"""
Application Setup & Bootstrap Module
Centralizes initialization, dependency injection, and configuration for the trading bot.
"""
import os
import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


def init_logging(level: str = "INFO", format_type: str = "standard") -> None:
    """Initialize application-wide logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: "standard" for human-readable, "json" for structured
    """
    # Clear any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure format
    if format_type.lower() == "json":
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Standard format compatible with existing print-based logs
        formatter = logging.Formatter('%(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def validate_environment() -> Dict[str, Any]:
    """Validate critical environment variables and return config dict.
    
    Returns:
        Dict with validated config values
        
    Raises:
        RuntimeError: If critical env vars are missing or invalid
    """
    config = {}
    
    # Required API keys
    finazon_key = os.getenv('FINAZON_API_KEY')
    if not finazon_key:
        raise RuntimeError("FINAZON_API_KEY not set in environment")
    config['finazon_api_key'] = finazon_key
    
    # Optional configurations with defaults
    config['trading_mode'] = os.getenv('TRADING_MODE', 'DRY').upper()
    config['rest_page_size'] = int(os.getenv('REST_PAGE_SIZE', '500'))
    config['model_db'] = os.getenv('MODEL_DB', 'modeldb.sqlite')
    
    # Validate trading mode
    if config['trading_mode'] not in ['DRY', 'LIVE']:
        raise RuntimeError(f"Invalid TRADING_MODE: {config['trading_mode']} (must be DRY or LIVE)")
    
    return config


def create_model_trainer():
    """Factory function to create and initialize the model trainer.
    
    Returns:
        ModelTrainer instance (started)
    """
    from .model_trainer import get_model_trainer
    return get_model_trainer()


def create_db_connection_factory():
    """Factory function to create database connection utilities.
    
    Returns:
        Dict with DB utility functions
    """
    from .data.database import open_conn, get_db_path, _init_db_pragmas
    
    return {
        'open_conn': open_conn,
        'get_db_path': get_db_path,
        'init_pragmas': _init_db_pragmas
    }


def create_market_api_adapter():
    """Factory function to create market API adapter.
    
    Returns:
        Market API adapter with normalized interface
    """
    from .api.kalshi import get_orderbook, create_order_compat
    
    class MarketAPIAdapter:
        """Normalized interface for market operations"""
        
        @staticmethod
        def get_orderbook(ticker: str) -> Optional[Dict]:
            """Get orderbook for a ticker"""
            return get_orderbook(ticker)
        
        @staticmethod
        def create_order(ticker: str, side: str, price_cents: int, qty: int, post_only: bool = True) -> Dict:
            """Create an order with normalized parameters"""
            return create_order_compat(
                ticker=ticker,
                side=side,
                price_cents=price_cents,
                qty=qty,
                post_only=post_only
            )
    
    return MarketAPIAdapter()


def create_predictor_factory():
    """Factory function to create prediction utilities.
    
    Returns:
        Predictor factory functions
    """
    try:
        from .models.predictor import get_fast_predictor
        return {'get_fast_predictor': get_fast_predictor}
    except ImportError:
        # Fallback if predictor module structure is different
        try:
            from .predictor import get_fast_predictor
            return {'get_fast_predictor': get_fast_predictor}
        except ImportError:
            logging.warning("Could not import predictor module")
            return {}


def create_ingestion_workers():
    """Factory function to create data ingestion worker functions.
    
    Returns:
        Dict with worker start functions
    """
    from .data.ingestion import start_backfill_worker, start_ws_listener
    
    return {
        'start_backfill_worker': start_backfill_worker,
        'start_ws_listener': start_ws_listener
    }


def create_trading_workers():
    """Factory function to create trading worker functions.
    
    Returns:
        Dict with trading worker functions
    """
    from .trading.strategy import start_trade_worker
    
    return {
        'start_trade_worker': start_trade_worker
    }


def bootstrap_application() -> Dict[str, Any]:
    """Bootstrap the entire application with all dependencies.
    
    Returns:
        Dict containing all initialized components
        
    Raises:
        RuntimeError: If critical initialization fails
    """
    # Initialize logging first
    init_logging()
    
    # Validate environment
    config = validate_environment()
    
    # Initialize database
    db_factory = create_db_connection_factory()
    db_factory['init_pragmas']()
    
    # Create core components
    model_trainer = create_model_trainer()
    market_api = create_market_api_adapter()
    predictor_factory = create_predictor_factory()
    ingestion_workers = create_ingestion_workers()
    trading_workers = create_trading_workers()
    
    return {
        'config': config,
        'model_trainer': model_trainer,
        'market_api': market_api,
        'predictor_factory': predictor_factory,
        'ingestion_workers': ingestion_workers,
        'trading_workers': trading_workers,
        'db_factory': db_factory
    }


def create_test_environment() -> Dict[str, Any]:
    """Create a test environment for unit/integration tests.
    
    Returns:
        Dict with test-friendly component instances
    """
    # Use minimal logging for tests
    init_logging(level="WARNING")
    
    # Mock environment for tests
    test_config = {
        'finazon_api_key': 'test_key',
        'trading_mode': 'DRY',
        'rest_page_size': 100,
        'model_db': ':memory:'
    }
    
    # Create minimal components for testing
    return {
        'config': test_config,
        'model_trainer': None,  # Can be mocked in tests
        'market_api': None,     # Can be mocked in tests
        'predictor_factory': {},
        'db_factory': create_db_connection_factory()
    }


def health_check() -> Dict[str, str]:
    """Perform basic health checks on application components.
    
    Returns:
        Dict with component status ("ok", "warning", "error")
    """
    status = {}
    
    # Check environment
    try:
        validate_environment()
        status['environment'] = 'ok'
    except Exception as e:
        status['environment'] = f'error: {e}'
    
    # Check database
    try:
        db_factory = create_db_connection_factory()
        conn = db_factory['open_conn'](read_only=True)
        conn.close()
        status['database'] = 'ok'
    except Exception as e:
        status['database'] = f'error: {e}'
    
    # Check model trainer
    try:
        trainer = create_model_trainer()
        if trainer.is_running:
            status['model_trainer'] = 'ok'
        else:
            status['model_trainer'] = 'warning: not running'
    except Exception as e:
        status['model_trainer'] = f'error: {e}'
    
    return status
