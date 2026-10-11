"""Persistence and inference helpers for the fitted MedSignal model."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS

ARTIFACT_VERSION = 1
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "model_pipeline.joblib"
)


def save_model_artifact(
    model: Pipeline,
    metrics: dict[str, Any],
    path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Save the fitted preprocessing-and-model pipeline and evaluation metadata."""
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("The model must be fitted on named feature columns.")
    if list(model.feature_names_in_) != list(FEATURE_COLUMNS):
        raise ValueError("The fitted model feature schema does not match MedSignal.")

    artifact_path = Path(path)
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model": model,
        "metrics": metrics,
        "features": list(FEATURE_COLUMNS),
    }
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, artifact_path)
    except OSError as error:
        raise RuntimeError(
            f"Could not save the trained model artifact to '{artifact_path}'."
        ) from error
    return artifact_path


def load_model_artifact(
    path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Load and validate a previously saved MedSignal model artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at '{artifact_path}'. "
            "Run `python train_model.py` to train and save it."
        )

    try:
        artifact = joblib.load(artifact_path)
    except (OSError, EOFError, ValueError) as error:
        raise RuntimeError(
            f"Could not read the model artifact at '{artifact_path}'. "
            "Run `python train_model.py` to create a fresh artifact."
        ) from error

    if not isinstance(artifact, dict):
        raise RuntimeError("The model artifact has an invalid format.")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError("The model artifact version is unsupported.")
    if artifact.get("features") != list(FEATURE_COLUMNS):
        raise RuntimeError("The model artifact feature schema does not match MedSignal.")
    if not isinstance(artifact.get("model"), Pipeline):
        raise RuntimeError("The model artifact does not contain a Scikit-learn pipeline.")
    if not isinstance(artifact.get("metrics"), dict):
        raise RuntimeError("The model artifact does not contain evaluation metrics.")

    model = artifact["model"]
    if list(getattr(model, "feature_names_in_", [])) != list(FEATURE_COLUMNS):
        raise RuntimeError("The fitted model's input schema does not match MedSignal.")
    return artifact


def predict_with_artifact(
    artifact: dict[str, Any],
    inputs: pd.DataFrame,
) -> dict[str, float | int]:
    """Predict a binary label and estimated class probabilities for feature rows."""
    if list(inputs.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Prediction inputs must contain exactly the supported MedSignal "
            "features in the expected order."
        )
    if inputs.empty:
        raise ValueError("Provide at least one input row for prediction.")

    model = artifact["model"]
    prediction = int(model.predict(inputs)[0])
    probabilities = model.predict_proba(inputs)
    positive_class_positions = np.flatnonzero(model.classes_ == 1)
    if positive_class_positions.size != 1:
        raise RuntimeError("The trained model does not provide disease probabilities.")
    probability_present = float(probabilities[0, positive_class_positions[0]])

    return {
        "prediction": prediction,
        "probability_no_disease": 1.0 - probability_present,
        "probability_disease_present": probability_present,
    }
