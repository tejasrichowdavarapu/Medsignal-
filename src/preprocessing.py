"""Leakage-safe preprocessing and baseline model construction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORICAL_FEATURES = (
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Create transformations that are fitted on training data only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def create_model_pipeline() -> Pipeline:
    """Build the logistic-regression baseline with preprocessing in one pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
