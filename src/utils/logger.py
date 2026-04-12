from __future__ import annotations

from pathlib import Path
import logging


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "network_sentinel.log"


def get_logger(name: str) -> logging.Logger:
    """
    Create or return a shared project logger.
    All modules log into the same file for easier debugging.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger