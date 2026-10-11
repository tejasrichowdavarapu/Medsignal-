"""Model-coefficient explanations for the MedSignal logistic-regression baseline."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS


def _coefficient_groups(model: Pipeline) -> tuple[np.ndarray, list[str]]:
    if not isinstance(model, Pipeline):
        raise ValueError("Feature insights require a fitted Scikit-learn pipeline.")
    if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("The model pipeline is missing required named steps.")
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    if not hasattr(classifier, "coef_") or not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError("Feature insights require a fitted linear classifier.")

    encoded_names = preprocessor.get_feature_names_out()
    coefficients = np.asarray(classifier.coef_)
    if coefficients.shape != (1, len(encoded_names)):
        raise ValueError("Expected binary Logistic Regression coefficients.")

    source_features = []
    for encoded_name in encoded_names:
        matching = next(
            (
                feature
                for feature in FEATURE_COLUMNS
                if (
                    encoded_name == f"numeric__{feature}"
                    or encoded_name.startswith(f"categorical__{feature}_")
                )
            ),
            None,
        )
        if matching is None:
            raise ValueError(
                f"Could not map encoded feature '{encoded_name}' to the input schema."
            )
        source_features.append(matching)
    return coefficients[0], source_features


def global_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Group absolute logistic-regression coefficients by their input feature."""
    coefficients, source_features = _coefficient_groups(model)
    grouped = {
        feature: float(
            np.linalg.norm(
                coefficients[np.asarray(source_features, dtype=object) == feature]
            )
        )
        for feature in FEATURE_COLUMNS
    }
    return (
        pd.DataFrame(
            [{"feature": name, "importance": value} for name, value in grouped.items()]
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def local_feature_contributions(
    model: Pipeline,
    inputs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute grouped feature contributions to the binary logistic score."""
    if list(inputs.columns) != list(FEATURE_COLUMNS) or len(inputs) != 1:
        raise ValueError("Provide one input row with the exact MedSignal feature schema.")
    coefficients, source_features = _coefficient_groups(model)
    transformed = model.named_steps["preprocessor"].transform(inputs)
    if issparse(transformed):
        transformed = transformed.toarray()
    contributions = np.asarray(transformed)[0] * coefficients
    grouped = {
        feature: float(
            contributions[np.asarray(source_features, dtype=object) == feature].sum()
        )
        for feature in FEATURE_COLUMNS
    }
    return (
        pd.DataFrame(
            [{"feature": name, "contribution": value} for name, value in grouped.items()]
        )
        .sort_values("contribution", key=lambda values: values.abs(), ascending=False)
        .reset_index(drop=True)
    )
