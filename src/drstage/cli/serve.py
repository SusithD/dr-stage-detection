"""Start the RetinaScan web app (API plus built frontend) with uvicorn."""

import argparse
import os
import sys

import uvicorn

from drstage.utils.logging import get_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options for the web server."""
    parser = argparse.ArgumentParser(description="Serve the RetinaScan DR screening web app")
    parser.add_argument("--checkpoint", default=None, help="trained checkpoint; omit for demo mode")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Start the RetinaScan API and frontend with uvicorn."""
    args = parse_args(argv)
    if args.checkpoint:
        os.environ["DRSTAGE_CHECKPOINT"] = args.checkpoint
    os.environ["DRSTAGE_DEVICE"] = args.device
    get_logger("drstage")

    from drstage.api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
