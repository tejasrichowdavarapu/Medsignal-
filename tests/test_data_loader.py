import pandas as pd
import pytest

from src.data_loader import (
    binarize_target,
    clean_heart_disease_data,
    inspect_heart_disease_data,
)


def test_cleaner_converts_question_mark_missing_values_without_imputation():
    data = pd.DataFrame({" AGE ": ["54", "62"], "num": ["0", "?"]})

    cleaned = clean_heart_disease_data(data)

    assert cleaned.columns.tolist() == ["age", "num"]
    assert cleaned["age"].tolist() == [54, 62]
    assert pd.isna(cleaned.loc[1, "num"])


def test_cleaner_rejects_unexpected_text_in_numeric_features():
    data = pd.DataFrame({"age": ["adult"], "num": [0]})

    with pytest.raises(ValueError, match="non-numeric values"):
        clean_heart_disease_data(data)


def test_binarize_target_preserves_original_disease_presence_meaning():
    binary_target = binarize_target(pd.Series([0, 1, 2, 3, 4], name="num"))

    assert binary_target.tolist() == [0, 1, 1, 1, 1]
    assert binary_target.name == "target"


def test_binarize_target_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Expected UCI target labels"):
        binarize_target(pd.Series([0, 5], name="num"))


def test_inspection_reports_feature_missingness_and_class_distribution():
    data = pd.DataFrame(
        {
            "age": [54, 62, 58],
            "ca": [0, None, 1],
            "num": [0, 1, 1],
        }
    )

    summary = inspect_heart_disease_data(data)

    assert summary["rows"] == 3
    assert summary["feature_count"] == 2
    assert summary["features"] == ["age", "ca"]
    assert summary["missing_values"]["ca"] == 1
    assert summary["target_distribution"] == {
        "0": {"count": 1, "percent": 33.33},
        "1": {"count": 2, "percent": 66.67},
    }
