"""Train and evaluate the initial MedSignal baseline model."""

import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import binarize_target, load_heart_disease_data
from src.evaluate import evaluate_binary_classifier
from src.model_artifact import DEFAULT_ARTIFACT_PATH, save_model_artifact
from src.preprocessing import FEATURE_COLUMNS, create_model_pipeline

RANDOM_SEED = 42
TEST_SIZE = 0.2


def train_and_evaluate() -> tuple[Pipeline, dict[str, object]]:
    """Split the data, fit only on training rows, and evaluate the held-out rows."""
    data = load_heart_disease_data()
    features = data.loc[:, FEATURE_COLUMNS]
    target = binarize_target(data["num"])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    model = create_model_pipeline()
    model.fit(x_train, y_train)
    metrics = evaluate_binary_classifier(model, x_test, y_test)

    results = {
        "model": "Logistic Regression",
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "training_samples": int(len(x_train)),
        "features": list(FEATURE_COLUMNS),
        **metrics,
    }
    return model, results


def main() -> None:
    model, results = train_and_evaluate()
    artifact_path = save_model_artifact(model, results)
    print("MedSignal baseline evaluation (held-out test set)")
    print(json.dumps(results, indent=2))
    print(f"\nSaved fitted pipeline and evaluation metrics to: {artifact_path}")
    print(
        "\nThese metrics describe one small, stratified test split of this dataset. "
        "They are not clinical validation."
    )


if __name__ == "__main__":
    main()
