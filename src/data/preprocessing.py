from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger


logger = get_logger("network_sentinel_preprocessing")

REQUIRED_FEATURE_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Duration",
]

TARGET_COLUMN = "Label"

# Conservative clipping bound to prevent overflow/scaler failure
NUMERIC_CLIP_BOUND = 1e15


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw network traffic data.

    Steps:
    - strip column names
    - remove duplicates
    - fill missing Flow Bytes/s with 0
    - replace +/- inf with 0
    - coerce numeric columns safely
    - clip extreme numeric values
    """
    try:
        cleaned_df = df.copy()

        logger.info("Starting data cleaning. Original shape: %s", cleaned_df.shape)

        cleaned_df.columns = cleaned_df.columns.str.strip()
        cleaned_df = cleaned_df.drop_duplicates()

        if "Flow Bytes/s" in cleaned_df.columns:
            missing_count = cleaned_df["Flow Bytes/s"].isna().sum()
            logger.info("Missing values in 'Flow Bytes/s': %s", missing_count)
            cleaned_df["Flow Bytes/s"] = cleaned_df["Flow Bytes/s"].fillna(0)

        feature_cols = [col for col in cleaned_df.columns if col != TARGET_COLUMN]

        for col in feature_cols:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

        cleaned_df[feature_cols] = cleaned_df[feature_cols].replace([np.inf, -np.inf], 0)
        cleaned_df[feature_cols] = cleaned_df[feature_cols].fillna(0)
        cleaned_df[feature_cols] = cleaned_df[feature_cols].clip(
            lower=-NUMERIC_CLIP_BOUND,
            upper=NUMERIC_CLIP_BOUND,
        )

        logger.info("Data cleaning completed. Cleaned shape: %s", cleaned_df.shape)
        return cleaned_df

    except Exception as exc:
        logger.exception("Failed during clean_data.")
        raise RuntimeError(f"Failed to clean data: {exc}") from exc


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features used in the project.
    """
    try:
        engineered_df = df.copy()

        missing_columns = [
            col for col in REQUIRED_FEATURE_COLUMNS if col not in engineered_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for feature engineering: {missing_columns}"
            )

        engineered_df["Total Packets"] = (
            engineered_df["Total Fwd Packets"] + engineered_df["Total Backward Packets"]
        )

        engineered_df["Packets per Second"] = (
            engineered_df["Total Packets"] / (engineered_df["Flow Duration"] + 1)
        )

        logger.info("Feature engineering completed successfully.")
        return engineered_df

    except Exception as exc:
        logger.exception("Failed during engineer_features.")
        raise RuntimeError(f"Failed to engineer features: {exc}") from exc


def final_numeric_sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup after feature engineering and before scaling/inference.
    """
    try:
        checked_df = df.copy()
        feature_cols = [col for col in checked_df.columns if col != TARGET_COLUMN]

        for col in feature_cols:
            checked_df[col] = pd.to_numeric(checked_df[col], errors="coerce")

        checked_df[feature_cols] = checked_df[feature_cols].replace([np.inf, -np.inf], 0)
        checked_df[feature_cols] = checked_df[feature_cols].fillna(0)
        checked_df[feature_cols] = checked_df[feature_cols].clip(
            lower=-NUMERIC_CLIP_BOUND,
            upper=NUMERIC_CLIP_BOUND,
        )

        numeric_array = checked_df[feature_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric_array).all():
            raise ValueError("Non-finite values still present after final numeric sanity check.")

        logger.info("Final numeric sanity check passed.")
        return checked_df

    except Exception as exc:
        logger.exception("Failed during final_numeric_sanity_check.")
        raise RuntimeError(f"Failed final numeric sanity check: {exc}") from exc


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(
            "Split features and target successfully. X shape: %s, y shape: %s",
            X.shape,
            y.shape,
        )
        return X, y

    except Exception as exc:
        logger.exception("Failed during split_features_and_target.")
        raise RuntimeError(f"Failed to split features and target: {exc}") from exc


def encode_target(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        logger.info("Target encoding completed. Classes: %s", list(label_encoder.classes_))
        return y_encoded, label_encoder

    except Exception as exc:
        logger.exception("Failed during encode_target.")
        raise RuntimeError(f"Failed to encode target labels: {exc}") from exc


def fit_scaler(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    try:
        X_numeric = X.to_numpy(dtype=np.float64)

        if not np.isfinite(X_numeric).all():
            raise ValueError("Input to scaler still contains non-finite values.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)

        if not np.isfinite(X_scaled).all():
            raise ValueError("Scaled features contain non-finite values.")

        logger.info("Feature scaling completed.")
        return X_scaled, scaler

    except Exception as exc:
        logger.exception("Failed during fit_scaler.")
        raise RuntimeError(f"Failed to fit scaler: {exc}") from exc


def transform_features(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    try:
        X_numeric = X.to_numpy(dtype=np.float64)

        if not np.isfinite(X_numeric).all():
            raise ValueError("Inference features contain non-finite values before scaling.")

        X_scaled = scaler.transform(X_numeric)

        if not np.isfinite(X_scaled).all():
            raise ValueError("Inference features contain non-finite values after scaling.")

        logger.info("Feature transformation completed for inference.")
        return X_scaled

    except Exception as exc:
        logger.exception("Failed during transform_features.")
        raise RuntimeError(f"Failed to transform features: {exc}") from exc


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder, list[str]]:
    try:
        prepared_df = clean_data(df)
        prepared_df = engineer_features(prepared_df)
        prepared_df = final_numeric_sanity_check(prepared_df)

        X, y = split_features_and_target(prepared_df, target_column=target_column)
        y_encoded, label_encoder = encode_target(y)
        X_scaled, scaler = fit_scaler(X)

        feature_columns = X.columns.tolist()

        logger.info("Training data prepared successfully.")
        return X_scaled, y_encoded, scaler, label_encoder, feature_columns

    except Exception as exc:
        logger.exception("Failed during prepare_training_data.")
        raise RuntimeError(f"Failed to prepare training data: {exc}") from exc


def prepare_single_input_for_inference(
    input_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    try:
        prepared_df = clean_data(input_df)
        prepared_df = engineer_features(prepared_df)
        prepared_df = final_numeric_sanity_check(prepared_df)

        if TARGET_COLUMN in prepared_df.columns:
            prepared_df = prepared_df.drop(columns=[TARGET_COLUMN])

        if feature_columns is not None:
            for feature in feature_columns:
                if feature not in prepared_df.columns:
                    prepared_df[feature] = 0

            extra_columns = [col for col in prepared_df.columns if col not in feature_columns]
            if extra_columns:
                logger.info("Dropping extra inference columns: %s", extra_columns)

            prepared_df = prepared_df[feature_columns]

        logger.info("Inference input prepared successfully. Shape: %s", prepared_df.shape)
        return prepared_df

    except Exception as exc:
        logger.exception("Failed during prepare_single_input_for_inference.")
        raise RuntimeError(f"Failed to prepare inference input: {exc}") from exc