"""Logging utilities with idempotent handler registration."""

import logging
from pathlib import Path


def get_logger(
    name: str, log_file: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Get or create a logger with idempotent handlers.

    Calling twice with the same name does not duplicate handlers.

    Args:
        name: Logger name.
        log_file: Optional file path for file handler.
        level: Logging level (default INFO).

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
