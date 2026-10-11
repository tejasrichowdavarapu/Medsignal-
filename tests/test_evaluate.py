import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_binary_classifier


def test_evaluation_returns_real_binary_metrics_and_ordered_confusion_matrix():
    features = pd.DataFrame({"signal": [0, 1, 2, 3, 4, 5]})
    target = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(features, target)

    result = evaluate_binary_classifier(model, features, target)

    assert set(result) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "test_samples",
    }
    assert result["confusion_matrix"] == [[3, 0], [0, 3]]
    assert result["test_samples"] == 6
    assert all(0.0 <= result[metric] <= 1.0 for metric in (
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ))


def test_evaluation_rejects_one_class_test_target():
    features = pd.DataFrame({"signal": [0, 1, 2, 3]})
    target = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(features, target)

    with pytest.raises(ValueError, match="must contain both classes"):
        evaluate_binary_classifier(model, features, np.array([0, 0]))
