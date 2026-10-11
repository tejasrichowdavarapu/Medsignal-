import numpy as np
import pandas as pd
import pytest

from src.insights import global_feature_importance, local_feature_contributions
from src.preprocessing import FEATURE_COLUMNS, create_model_pipeline


def make_fitted_model():
    features = pd.DataFrame(
        {
            column: np.arange(30, dtype=float) + index
            for index, column in enumerate(FEATURE_COLUMNS)
        }
    )
    features["sex"] = np.tile([0, 1], 15)
    features["cp"] = np.tile([1, 2, 3, 4, 1, 2], 5)
    features["fbs"] = np.tile([0, 1], 15)
    features["restecg"] = np.tile([0, 1, 2], 10)
    features["exang"] = np.tile([0, 1], 15)
    features["slope"] = np.tile([1, 2, 3], 10)
    features["ca"] = np.tile([0, 1, 2, 3, 0], 6)
    features["thal"] = np.tile([3, 6, 7], 10)
    target = pd.Series(np.tile([0, 1], 15))
    return create_model_pipeline().fit(features, target), features.iloc[[0]]


def test_global_feature_importance_covers_all_input_features():
    model, _ = make_fitted_model()

    importance = global_feature_importance(model)

    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert (importance["importance"] >= 0).all()


def test_local_contributions_sum_to_the_logistic_decision_score():
    model, sample = make_fitted_model()

    contributions = local_feature_contributions(model, sample)

    assert set(contributions["feature"]) == set(FEATURE_COLUMNS)
    assert contributions["contribution"].sum() == pytest.approx(
        model.decision_function(sample)[0] - model.named_steps["classifier"].intercept_[0]
    )


def test_local_contributions_reject_mismatched_input_schema():
    model, sample = make_fitted_model()

    with pytest.raises(ValueError, match="exact MedSignal feature schema"):
        local_feature_contributions(model, sample.drop(columns=["age"]))
