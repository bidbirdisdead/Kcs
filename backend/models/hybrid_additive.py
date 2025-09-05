import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit


class HybridAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    y_hat = lin.predict(X) + gamma * gbm.predict(X)
    - Fit linear (ridge/huber) on y
    - Fit LightGBM on residuals r = y - y_lin
    - Choose gamma ∈ [0,1] via time-series CV (MAE by default)
    - If GBM is degenerate (no trees / zero importances), gamma=0
    - SAFE IMPORT: lightgbm is imported inside _make_gbm() so module import won't crash if LGBM isn't installed.
    """
    def __init__(self, linear="huber", ridge_alpha=0.5, huber_epsilon=1.35,
                 gbm_params=None, n_estimators=1500, learning_rate=0.05,
                 min_child_samples=16, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                 random_state=42, cv_splits=4, gamma_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
                 score="mae"):
        self.linear = linear
        self.ridge_alpha = ridge_alpha
        self.huber_epsilon = huber_epsilon
        self.gbm_params = gbm_params or {}
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # Use LightGBM canonical name internally. Accept legacy param name in callers
        # by keeping compatibility in the calling sites; internally prefer
        # `min_child_samples` to avoid LightGBM alias warnings.
        self.min_child_samples = min_child_samples
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.gamma_grid = gamma_grid
        self.score = score
        self.lin_ = None
        self.gbm_ = None
        self.gamma_ = 0.0
        self.degenerate_gbm_ = False

    def _make_linear(self):
        if self.linear == "ridge":
            return Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        return HuberRegressor(epsilon=self.huber_epsilon, alpha=self.ridge_alpha, fit_intercept=True)

    def _make_gbm(self):
        # Import inside to avoid module import errors when USE_HYBRID=0 (or LGBM not installed)
        import lightgbm as lgb
        params = dict(
            objective="regression_l1",         # robust for tiny heavy-tailed residuals
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            min_split_gain=0.0,
        )
        params.update(self.gbm_params)
        return lgb.LGBMRegressor(**params)

    def _train_lgbm(self, Xg, yg):
        """Train LGBM with adaptive min_child_samples and clear RuntimeError on degeneracy."""
        import lightgbm as lgb
        Xg = np.asarray(Xg, dtype=float)
        yg = np.asarray(yg, dtype=float)
        n = Xg.shape[0]
        p = Xg.shape[1] if Xg.ndim > 1 else 1

        def _too_flat(y, eps=1e-9):
            return float(np.nanstd(y)) < eps

        def _lowuniq_cols(X, k=2):
            try:
                return sum(len(np.unique(X[:, j])) < k for j in range(X.shape[1]))
            except Exception:
                return 0

        print(f"[LGBM][DIAG][hybrid] n={n} p={p} y_std={float(np.nanstd(yg)):.3e} lowuniq={_lowuniq_cols(Xg)}")

        if n < 200:
            raise RuntimeError(f"LGBM degenerate: too_few_rows n={n} (<200)")
        if _too_flat(yg) or _lowuniq_cols(Xg) == p:
            raise RuntimeError("LGBM degenerate: flat target or all columns low-unique")

        mcs = max(8, int(0.005 * n))

        gbm = self._make_gbm()
        # ensure configured min_child_samples is overridden by adaptive mcs
        try:
            setattr(gbm, 'min_child_samples', mcs)
        except Exception:
            pass

        split = int(n * 0.85)
        if split >= 200 and (n - split) >= 50:
            callbacks = [lgb.early_stopping(80, verbose=False)]
            gbm.fit(Xg[:split], yg[:split], eval_set=[(Xg[split:], yg[split:])], eval_metric='l1', callbacks=callbacks)
        else:
            gbm.fit(Xg, yg)

        try:
            nt = gbm.booster_.num_trees()
        except Exception:
            nt = -1
        if nt <= 0:
            raise RuntimeError('LGBM degenerate: no_usable_splits (num_trees==0)')
        return gbm

    def _should_fit_gbm(self, X, r):
        """Return True if GBM should be fit on residuals r with features X.

        Avoid fitting when residuals or features are degenerate (constant,
        NaN/inf, or too few samples relative to min_data_in_leaf). This
        prevents LightGBM from emitting repeated "No further splits" warnings
        on pathological inputs.
    """
    # Basic sanity
        if X is None or r is None:
            return False
        X = np.asarray(X)
        r = np.asarray(r).ravel()
        n = len(r)
        if n < 3:
            return False

        # NaN/Inf check
        if not np.isfinite(r).all() or not np.isfinite(X).all():
            return False

        # Residual variance too small (effectively constant)
        if float(np.nanstd(r)) <= 1e-12 or np.allclose(r, r[0]):
            return False

        # Feature variance: at least one feature must vary
        col_std = np.nanstd(X, axis=0)
        if np.all(col_std <= 1e-12):
            return False

        # Ensure enough samples relative to configured minimum child samples
        min_leaf = int(self.min_child_samples) if self.min_child_samples is not None else 1
        # require at least 3 * min_leaf samples to allow splits
        if n < max(3, min_leaf * 3):
            return False

        return True

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).ravel()

        # 1) Linear baseline
        self.lin_ = self._make_linear()
        self.lin_.fit(X, y)
        y_lin = self.lin_.predict(X)
        r = y - y_lin

        # 2) GBM on residuals (with simple early stop on last 15% if n>=200)
        # Decide whether we should attempt to fit a GBM at all. If inputs are
        # degenerate, avoid fitting to prevent LightGBM warnings like
        # "No further splits with positive gain" and mark as degenerate.
        self.gbm_ = None
        try:
            if not self._should_fit_gbm(X, r):
                self.gbm_ = None
            else:
                n = len(y)
                try:
                    if n >= 200:
                        split = int(n * 0.85)
                        if self._should_fit_gbm(X[:split], r[:split]) and self._should_fit_gbm(X[split:], r[split:]):
                            gbm = self._train_lgbm(X, r)
                        else:
                            gbm = None
                    else:
                        if self._should_fit_gbm(X, r):
                            gbm = self._train_lgbm(X, r)
                        else:
                            gbm = None
                except RuntimeError:
                    gbm = None
                self.gbm_ = gbm
        except Exception:
            # If LightGBM raises, treat as degenerate and avoid bubbling exceptions
            self.gbm_ = None

        # Degenerate detection
        self.degenerate_gbm_ = False
        if self.gbm_ is None:
            self.degenerate_gbm_ = True
        else:
            try:
                if self.gbm_.booster_.num_trees() == 0:
                    self.degenerate_gbm_ = True
            except Exception:
                fi = getattr(self.gbm_, "feature_importances_", None)
                if fi is None or float(np.sum(fi)) == 0.0:
                    self.degenerate_gbm_ = True

        # 3) gamma via time-series CV
        if self.degenerate_gbm_:
            self.gamma_ = 0.0
            return self

        tscv = TimeSeriesSplit(n_splits=min(self.cv_splits, max(2, len(y) // 100)))
        best_gamma, best_err = 0.0, float("inf")
        for g in self.gamma_grid:
            errs = []
            for tr, va in tscv.split(X):
                Xtr, Xva = X[tr], X[va]; ytr, yva = y[tr], y[va]
                lin = clone(self.lin_); lin.fit(Xtr, ytr)
                rtr = ytr - lin.predict(Xtr)
                # In CV folds, only fit GBM when the training fold has
                # sufficient signal; otherwise treat gbm prediction as zero.
                gbm = None
                if self._should_fit_gbm(Xtr, rtr):
                    try:
                        gbm = self._make_gbm()
                        gbm.fit(Xtr, rtr)
                    except Exception:
                        gbm = None

                yhat = lin.predict(Xva)
                if gbm is not None and g != 0.0:
                    # safe: gbm.predict may warn if features mismatch; catch broadly
                    try:
                        yhat = yhat + g * gbm.predict(Xva)
                    except Exception:
                        # fallback: treat gbm contribution as zero for this fold
                        pass
                if self.score == "mae":
                    errs.append(np.mean(np.abs(yva - yhat)))
                else:
                    d = yva - yhat; errs.append(np.mean(d * d))
            avg = float(np.mean(errs)) if errs else float("inf")
            if avg < best_err:
                best_err, best_gamma = avg, g
        self.gamma_ = best_gamma
        return self

    def predict(self, X):
        y_lin = self.lin_.predict(X)
        if self.gbm_ is None or self.gamma_ == 0.0:
            return y_lin
        return y_lin + self.gamma_ * self.gbm_.predict(X)

