from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger("network_sentinel_loader")


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.
    """
    try:
        logger.info("Loading dataset from path: %s", path)
        df = pd.read_csv(path, low_memory=False)
        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
        return df

    except FileNotFoundError as exc:
        logger.exception("Dataset file not found.")
        raise FileNotFoundError(f"Dataset file not found: {path}") from exc

    except Exception as exc:
        logger.exception("Unexpected error while loading dataset.")
        raise RuntimeError(f"Failed to load dataset from {path}: {exc}") from exc