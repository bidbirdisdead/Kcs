# Backend Module

## Overview
The backend module contains the core trading bot functionality with a clean, modular architecture.

## Module Structure

### Core Modules
- `app_setup.py` - Application bootstrap, dependency injection, and configuration
- `main.py` - Entry point and orchestration
- `config.py` - Configuration constants and environment settings

### Domain Modules
- `data/` - Data ingestion, database operations, and persistence
  - `database.py` - Database utilities and connection management
  - `ingestion.py` - Price data backfill and WebSocket listeners
- `models/` - Machine learning models and prediction
  - `trainer.py` - Model training pipeline 
  - `predictor.py` - Fast prediction interface
- `trading/` - Trading logic and execution
  - `strategy.py` - ML-based arbitrage strategy
- `api/` - External API integrations
  - `kalshi.py` - Kalshi API client
  - `finazon.py` - Finazon data API client
- `features/` - Feature engineering for ML models

## Usage

### Running the Bot
```bash
python backend/main.py
```

### Environment Variables
- `FINAZON_API_KEY` - Required for price data
- `TRADING_MODE` - "DRY" or "LIVE" (default: DRY)
- `REST_PAGE_SIZE` - API pagination size (default: 500)
- `ENABLE_MICRO_FEATS` - Set to "1" to enable microstructure features (OBI, WMP, Spread) in the predictor. (default: 0)

#### Kalshi Environment
- `KALSHI_API_BASE`: Base URL for Kalshi REST (default: `https://api.elections.kalshi.com/trade-api/v2`).
- `KALSHI_API_KEY` / `KALSHI_API_SECRET`: Credentials used for REST and WS auth where applicable.
- `KALSHI_ENV`: `PROD` or `DEMO` (default empty). When `DEMO`, and `KALSHI_DEMO_BASE` is set, GET requests use the demo base.
- `KALSHI_DEMO_BASE`: Optional demo REST base URL.

Notes:
- REST and WebSocket endpoints use different schemes/hosts. REST uses `https://.../trade-api/v2`, while WS endpoints are `wss://...` on the elections domain. This repo’s user-fills WS defaults to `wss://api.elections.kalshi.com/ws/user-fills`.
- In `TRADING_MODE=DRY`, GET endpoints (e.g., orderbooks/markets) are allowed and prefer live REST when configured. Mutating calls are skipped and return a dry-run payload.

### Testing
```bash
python test_app_setup.py  # Test application setup
pytest                    # Run full test suite
```

## Architecture Principles

### Separation of Concerns
- Each module has a single, well-defined responsibility
- Business logic is isolated from infrastructure concerns
- External dependencies are abstracted behind adapters

### Dependency Injection
- `app_setup.py` provides factory functions for all major components
- Components receive dependencies explicitly rather than importing globally
- Makes testing and mocking straightforward

### Error Handling
- Graceful degradation when external services are unavailable
- Clear error messages for configuration issues
- Health checks for monitoring component status

### Logging
- Centralized logging configuration in `app_setup.py`
- Structured logging support for production monitoring
- Consistent log message formatting across modules

## Key Components

### Model Trainer
- Background training pipeline for ML models
- Manages model lifecycle and freshness
- Provides readiness checks for trading logic

### Data Ingestion
- WebSocket real-time price feeds
- REST API backfill for missing data
- SQLite persistence with WAL mode

### Trading Strategy
- ML-based fair value calculation
- Kalshi arbitrage opportunity detection
- Dry-run and live trading modes

### Market API
- Normalized interface for market operations
- Order placement and orderbook access
- Rate limiting and error handling
