from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.data.loader import load_dataset
from src.data.preprocessing import prepare_training_data
from src.utils.logger import get_logger


logger = get_logger("network_sentinel_training")

MODEL_DIR = Path("models")
DATA_PATH = "data/raw/Wednesday-workingHours.pcap_ISCX.csv"


def load_and_prepare_data(data_path: str = DATA_PATH):
    """
    Load raw data and prepare it for model training.
    """
    try:
        logger.info("Loading and preparing data from: %s", data_path)
        df = load_dataset(data_path)

        X_scaled, y_encoded, scaler, label_encoder, feature_columns = prepare_training_data(df)

        logger.info(
            "Data prepared successfully. X shape: %s, y shape: %s",
            X_scaled.shape,
            y_encoded.shape,
        )

        return X_scaled, y_encoded, scaler, label_encoder, feature_columns

    except Exception as exc:
        logger.exception("Failed to load and prepare data.")
        raise RuntimeError(f"Failed to load and prepare data: {exc}") from exc


def split_data(X: np.ndarray, y: np.ndarray):
    """
    Split the dataset into train and test sets.
    """
    try:
        logger.info("Splitting data into train/test sets.")

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    except Exception as exc:
        logger.exception("Failed to split data.")
        raise RuntimeError(f"Failed to split data: {exc}") from exc


def build_models() -> dict[str, Any]:
    """
    Build all candidate models for comparison.
    """
    try:
        logger.info("Building candidate models.")

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
        }

        logger.info("Candidate models built successfully.")
        return models

    except Exception as exc:
        logger.exception("Failed to build candidate models.")
        raise RuntimeError(f"Failed to build candidate models: {exc}") from exc


def run_cross_validation(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict[str, Any]:
    """
    Run cross-validation for a single model.
    """
    try:
        logger.info("Running %s-fold cross-validation for model: %s", cv, model.__class__.__name__)

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

        return {
            "cv_scores": cv_scores,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        }

    except Exception as exc:
        logger.exception("Cross-validation failed for model: %s", model.__class__.__name__)
        raise RuntimeError(f"Cross-validation failed: {exc}") from exc


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, label_encoder) -> dict[str, Any]:
    """
    Evaluate a trained model on the held-out test set.
    """
    try:
        logger.info("Evaluating model: %s", model.__class__.__name__)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report_text = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        class_names = list(label_encoder.classes_)

        logger.info(
            "Evaluation complete for %s. Test accuracy: %.4f",
            model.__class__.__name__,
            accuracy,
        )

        return {
            "accuracy": float(accuracy),
            "classification_report": report_text,
            "confusion_matrix": cm,
            "class_names": class_names,
            "y_pred": y_pred,
        }

    except Exception as exc:
        logger.exception("Evaluation failed for model: %s", model.__class__.__name__)
        raise RuntimeError(f"Evaluation failed: {exc}") from exc


def compare_models(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder,
) -> tuple[Any, str, dict[str, Any]]:
    """
    Train, validate, and compare all models.
    Returns:
    - best fitted model
    - best model name
    - all results
    """
    try:
        logger.info("Starting model comparison.")

        all_results: dict[str, Any] = {}

        best_model = None
        best_model_name = None
        best_score = -1.0

        for model_name, model in models.items():
            logger.info("Processing model: %s", model_name)

            cv_results = run_cross_validation(model, X, y, cv=5)

            logger.info("Training model: %s", model_name)
            model.fit(X_train, y_train)

            evaluation_results = evaluate_model(model, X_test, y_test, label_encoder)

            all_results[model_name] = {
                "model": model,
                "cv_scores": cv_results["cv_scores"],
                "cv_mean": cv_results["cv_mean"],
                "cv_std": cv_results["cv_std"],
                "test_accuracy": evaluation_results["accuracy"],
                "classification_report": evaluation_results["classification_report"],
                "confusion_matrix": evaluation_results["confusion_matrix"],
                "class_names": evaluation_results["class_names"],
            }

            logger.info(
                "Model %s results: CV mean=%.4f, Test accuracy=%.4f",
                model_name,
                cv_results["cv_mean"],
                evaluation_results["accuracy"],
            )

            # Winner selection based on test accuracy first, then CV mean
            score = evaluation_results["accuracy"]
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = model_name

        if best_model is None or best_model_name is None:
            raise RuntimeError("No valid model was selected during comparison.")

        logger.info("Best model selected: %s", best_model_name)
        return best_model, best_model_name, all_results

    except Exception as exc:
        logger.exception("Model comparison failed.")
        raise RuntimeError(f"Model comparison failed: {exc}") from exc


def save_artifacts(model, scaler, label_encoder, feature_columns: list[str]) -> None:
    """
    Save all artifacts needed for inference.
    """
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_DIR / "model.pkl")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
        joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")

        logger.info("Artifacts saved successfully to %s", MODEL_DIR)

    except Exception as exc:
        logger.exception("Failed to save artifacts.")
        raise RuntimeError(f"Failed to save artifacts: {exc}") from exc


def print_model_comparison(results: dict[str, Any], winner_name: str) -> None:
    """
    Print comparison summary for all models.
    """
    print("\n=== Model Comparison Summary ===")

    for model_name, result in results.items():
        print(f"\n{model_name}")
        print(f"CV Scores: {np.round(result['cv_scores'], 4)}")
        print(f"Mean CV Accuracy: {result['cv_mean']:.4f}")
        print(f"CV Std Dev: {result['cv_std']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")

    print(f"\nSelected Best Model: {winner_name}")


def print_best_model_details(results: dict[str, Any], winner_name: str) -> None:
    """
    Print detailed evaluation for the winning model.
    """
    winner = results[winner_name]

    print("\n=== Winning Model Evaluation ===")
    print(f"Winning Model: {winner_name}")
    print(f"Test Accuracy: {winner['test_accuracy']:.4f}")

    print("\nClassification Report:\n")
    print(winner["classification_report"])

    print("Confusion Matrix:")
    print(winner["confusion_matrix"])

    print("\nSaved artifacts:")
    print("- models/model.pkl")
    print("- models/scaler.pkl")
    print("- models/label_encoder.pkl")
    print("- models/feature_columns.pkl")
    print("- logs/network_sentinel.log")


def train_model(data_path: str = DATA_PATH):
    """
    Full training pipeline:
    - load data
    - preprocess
    - split
    - build candidate models
    - compare candidates
    - save best model
    """
    try:
        logger.info("Starting full training pipeline.")

        X_scaled, y_encoded, scaler, label_encoder, feature_columns = load_and_prepare_data(data_path)

        X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded)

        models = build_models()

        best_model, best_model_name, results = compare_models(
            models=models,
            X=X_scaled,
            y=y_encoded,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder,
        )

        save_artifacts(best_model, scaler, label_encoder, feature_columns)

        print_model_comparison(results, best_model_name)
        print_best_model_details(results, best_model_name)

        logger.info("Training pipeline completed successfully.")
        return best_model

    except Exception as exc:
        logger.exception("Training pipeline failed.")
        raise RuntimeError(f"Unexpected error during model training: {exc}") from exc


if __name__ == "__main__":
    train_model()