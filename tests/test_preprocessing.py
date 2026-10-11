import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    create_model_pipeline,
)


def test_feature_groups_cover_the_expected_input_schema_once():
    assert len(FEATURE_COLUMNS) == 13
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_pipeline_fits_missing_values_and_predicts_from_all_features():
    features = pd.DataFrame(
        {
            column: np.arange(20, dtype=float) + index
            for index, column in enumerate(FEATURE_COLUMNS)
        }
    )
    features["sex"] = np.tile([0, 1], 10)
    features["cp"] = np.tile([1, 2, 3, 4], 5)
    features.loc[0, "chol"] = np.nan
    features.loc[1, "thal"] = np.nan
    target = pd.Series(np.tile([0, 1], 10))

    model = create_model_pipeline()
    model.fit(features, target)

    assert isinstance(model, Pipeline)
    assert model.predict(features).shape == (20,)
    assert model.predict_proba(features).shape == (20, 2)
