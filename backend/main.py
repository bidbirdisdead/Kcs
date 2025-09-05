import os
import sys
import time
import threading
import argparse
from dotenv import load_dotenv

# When this file is run directly (python backend/main.py), Python's import
# machinery may not recognize the package root. Fix sys.path so imports like
# 'backend.xxx' work whether the file is executed as a module or as a script.
if __package__ is None:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_this_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from backend.app_setup import bootstrap_application

def _run_all(components):
    config = components['config']
    ingestion_workers = components['ingestion_workers']
    trading_workers = components['trading_workers']

    print(f"FINAZON_API_KEY: {config['finazon_api_key']}")
    print("Starting trading bot...")

    # Start background workers
    threading.Thread(target=ingestion_workers['start_backfill_worker'], daemon=True).start()
    threading.Thread(target=trading_workers['start_trade_worker'], daemon=True).start()

    # Start WebSocket listener in the main thread (blocks)
    ingestion_workers['start_ws_listener']()

def main():
    load_dotenv()
    print(f"Current working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser(description="Kcs trading bot entrypoint")
    parser.add_argument(
        "--service",
        choices=["all", "backfill", "ws", "trainer", "trader"],
        default="all",
        help="Run a specific service in this process (default: all)",
    )
    args = parser.parse_args()

    # Bootstrap application with centralized setup
    try:
        components = bootstrap_application()

        if args.service == "all":
            _run_all(components)
            return

        if args.service == "backfill":
            print("[MAIN] Starting backfill worker (solo mode)")
            components['ingestion_workers']['start_backfill_worker']()
            return

        if args.service == "ws":
            print("[MAIN] Starting WebSocket listener (solo mode)")
            components['ingestion_workers']['start_ws_listener']()
            return

        if args.service == "trader":
            print("[MAIN] Starting trading worker (solo mode)")
            components['trading_workers']['start_trade_worker']()
            while True:
                time.sleep(60)

        if args.service == "trainer":
            print("[MAIN] Starting model trainer (solo mode)")
            # model trainer already runs in background; keep process alive
            _ = components['model_trainer']
            while True:
                time.sleep(60)
    
    except Exception as e:
        print(f"Failed to start trading bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
