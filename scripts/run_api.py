#!/usr/bin/env python3
"""
API server launcher for Medical AI Diagnosis System v2.0

Starts FastAPI server with uvicorn.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from utils.config import API_HOST, API_PORT, API_RELOAD
from utils.logger import api_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Medical AI Diagnosis API server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=API_HOST,
        help=f"Host address (default: {API_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=API_PORT,
        help=f"Port number (default: {API_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=API_RELOAD,
        help="Enable auto-reload on code changes",
    )

    args = parser.parse_args()

    api_logger.info("=" * 60)
    api_logger.info("MEDICAL AI DIAGNOSIS SYSTEM - API SERVER")
    api_logger.info("=" * 60)
    api_logger.info(f"Host: {args.host}")
    api_logger.info(f"Port: {args.port}")
    api_logger.info(f"Reload: {args.reload}")
    api_logger.info("=" * 60)
    api_logger.info("\nStarting server...")
    api_logger.info(f"API docs: http://{args.host}:{args.port}/docs")
    api_logger.info(f"Health check: http://{args.host}:{args.port}/health")
    api_logger.info("\nPress CTRL+C to stop\n")

    try:
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        api_logger.info("\nServer stopped by user")
    except Exception as e:
        api_logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
