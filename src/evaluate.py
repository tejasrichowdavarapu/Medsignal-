"""Evaluation helpers for held-out binary heart-disease predictions."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    model: Any,
    features: Any,
    target: Any,
) -> dict[str, Any]:
    """Compute standard metrics and a 0/1-ordered confusion matrix."""
    actual = np.asarray(target)
    if actual.ndim != 1 or len(actual) == 0:
        raise ValueError("Evaluation requires a non-empty one-dimensional target.")
    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Evaluation target must contain only binary labels 0 and 1.")
    if np.unique(actual).size != 2:
        raise ValueError("Evaluation target must contain both classes 0 and 1.")

    predicted = np.asarray(model.predict(features))
    if predicted.shape != actual.shape:
        raise ValueError("Model predictions must match the target shape.")
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Model predictions must contain only binary labels 0 and 1.")

    probabilities = np.asarray(model.predict_proba(features))
    model_classes = np.asarray(model.classes_)
    positive_class_positions = np.flatnonzero(model_classes == 1)
    if positive_class_positions.size != 1:
        raise ValueError("Model must provide probabilities for positive class 1.")
    positive_probabilities = probabilities[:, positive_class_positions[0]]

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, positive_probabilities)),
        "confusion_matrix": matrix.tolist(),
        "test_samples": int(actual.size),
    }
