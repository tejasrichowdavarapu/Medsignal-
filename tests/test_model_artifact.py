import numpy as np
import pandas as pd
import pytest

from src.model_artifact import (
    load_model_artifact,
    predict_with_artifact,
    save_model_artifact,
)
from src.preprocessing import FEATURE_COLUMNS, create_model_pipeline


def fitted_model():
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


def test_saved_artifact_round_trips_pipeline_metrics_and_predictions(tmp_path):
    model, sample = fitted_model()
    artifact_path = tmp_path / "models" / "pipeline.joblib"
    metrics = {"accuracy": 0.75, "confusion_matrix": [[2, 1], [0, 3]]}

    saved_path = save_model_artifact(model, metrics, artifact_path)
    loaded_artifact = load_model_artifact(saved_path)
    result = predict_with_artifact(loaded_artifact, sample)

    assert saved_path.is_file()
    assert loaded_artifact["metrics"] == metrics
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability_disease_present"] <= 1.0
    assert 0.0 <= result["probability_no_disease"] <= 1.0
    assert (
        result["probability_disease_present"] + result["probability_no_disease"]
        == pytest.approx(1.0)
    )
    assert result["prediction"] == int(model.predict(sample)[0])


def test_loading_missing_model_gives_a_training_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_model.py"):
        load_model_artifact(tmp_path / "missing.joblib")


def test_prediction_rejects_a_mismatched_feature_schema(tmp_path):
    model, sample = fitted_model()
    artifact_path = save_model_artifact(model, {}, tmp_path / "pipeline.joblib")
    artifact = load_model_artifact(artifact_path)

    with pytest.raises(ValueError, match="exactly the supported MedSignal features"):
        predict_with_artifact(artifact, sample.drop(columns=["age"]))
