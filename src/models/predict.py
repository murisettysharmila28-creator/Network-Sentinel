from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.preprocessing import prepare_single_input_for_inference, transform_features
from src.utils.logger import get_logger


logger = get_logger("network_sentinel_prediction")

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"


def _load_artifact(path: Path) -> Any:
    if not path.exists():
        logger.error("Required artifact not found: %s", path)
        raise FileNotFoundError(
            f"Required artifact not found: {path}. Train the model first."
        )
    return joblib.load(path)


def load_prediction_artifacts():
    """
    Load saved model artifacts required for inference.
    """
    try:
        logger.info("Loading prediction artifacts.")
        model = _load_artifact(MODEL_PATH)
        scaler = _load_artifact(SCALER_PATH)
        label_encoder = _load_artifact(LABEL_ENCODER_PATH)
        feature_columns = _load_artifact(FEATURE_COLUMNS_PATH)

        logger.info(
            "Prediction artifacts loaded successfully from %s",
            MODEL_DIR,
        )
        return model, scaler, label_encoder, feature_columns

    except Exception as exc:
        logger.exception("Failed to load prediction artifacts.")
        raise RuntimeError(f"Failed to load prediction artifacts: {exc}") from exc


def predict_attack(input_df: pd.DataFrame) -> str:
    """
    Predict the attack label for the first record in the input dataframe.
    """
    predicted_label, _ = predict_attack_with_confidence(input_df)
    return predicted_label


def predict_attack_with_confidence(input_df: pd.DataFrame) -> tuple[str, float]:
    """
    Predict attack label and confidence for the first record in input_df.
    """
    try:
        if input_df.empty:
            raise ValueError("Input dataframe is empty.")

        model, scaler, label_encoder, feature_columns = load_prediction_artifacts()

        logger.info("Inference input shape before preparation: %s", input_df.shape)

        prepared_df = prepare_single_input_for_inference(
            input_df=input_df,
            feature_columns=feature_columns,
        )

        logger.info("Prepared inference shape: %s", prepared_df.shape)

        X_scaled = transform_features(prepared_df, scaler)

        logger.info("Generating prediction.")
        predictions = model.predict(X_scaled)
        predicted_labels = label_encoder.inverse_transform(predictions)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)
            confidence = float(probabilities[0].max())
        else:
            confidence = 1.0

        predicted_label = str(predicted_labels[0])

        logger.info(
            "Prediction complete. Label=%s Confidence=%.4f",
            predicted_label,
            confidence,
        )
        return predicted_label, confidence

    except Exception as exc:
        logger.exception("Prediction failed.")
        raise RuntimeError(f"Unexpected error during prediction: {exc}") from exc


def predict_attack_batch(input_df: pd.DataFrame) -> dict[str, Any]:
    """
    Structured prediction wrapper for app/UI integration.

    Returns:
        {
            "predicted_attack": str,
            "confidence": float,
            "model_name": str,
            "class_probabilities": dict[str, float],
            "processed_df": pd.DataFrame
        }
    """
    try:
        if input_df.empty:
            raise ValueError("Input dataframe is empty.")

        model, scaler, label_encoder, feature_columns = load_prediction_artifacts()

        logger.info("Batch inference input shape before preparation: %s", input_df.shape)

        prepared_df = prepare_single_input_for_inference(
            input_df=input_df,
            feature_columns=feature_columns,
        )

        logger.info("Batch prepared inference shape: %s", prepared_df.shape)

        X_scaled = transform_features(prepared_df, scaler)

        logger.info("Generating batch prediction.")
        predictions = model.predict(X_scaled)
        predicted_labels = label_encoder.inverse_transform(predictions)

        class_probabilities: dict[str, float] = {}
        confidence = 1.0

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)
            first_row_probabilities = probabilities[0]
            confidence = float(first_row_probabilities.max())
            class_probabilities = {
                str(label): float(prob)
                for label, prob in zip(label_encoder.classes_, first_row_probabilities)
            }

        predicted_label = str(predicted_labels[0])

        logger.info(
            "Batch prediction complete. Label=%s Confidence=%.4f",
            predicted_label,
            confidence,
        )

        return {
            "predicted_attack": predicted_label,
            "confidence": confidence,
            "model_name": "XGBoost",
            "class_probabilities": class_probabilities,
            "processed_df": prepared_df,
        }

    except Exception as exc:
        logger.exception("Batch prediction failed.")
        raise RuntimeError(f"Unexpected error during batch prediction: {exc}") from exc