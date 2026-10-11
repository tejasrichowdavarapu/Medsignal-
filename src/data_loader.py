"""Loading, cleaning, and inspecting the UCI Heart Disease dataset."""

from pathlib import Path
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo


UCI_HEART_DISEASE_ID = 45
TARGET_COLUMN = "num"
TARGET_ENCODING = {
    0: "No angiographic heart disease",
    1: "Angiographic heart disease present (value 1)",
    2: "Angiographic heart disease present (value 2)",
    3: "Angiographic heart disease present (value 3)",
    4: "Angiographic heart disease present (value 4)",
}
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "uci_heart_disease.csv"
)


def clean_heart_disease_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and numeric values without imputing missing data."""
    if data.empty:
        raise ValueError("The heart-disease dataset is empty.")

    cleaned = data.copy()
    cleaned.columns = [str(column).strip().lower() for column in cleaned.columns]
    if cleaned.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")
    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"The dataset must contain the '{TARGET_COLUMN}' target column.")

    for column in cleaned.columns:
        values = cleaned[column]
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            values = values.astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})

        numeric_values = pd.to_numeric(values, errors="coerce")
        invalid_values = values.notna() & numeric_values.isna()
        if invalid_values.any():
            examples = values[invalid_values].astype(str).unique()[:3].tolist()
            raise ValueError(
                f"Column '{column}' contains non-numeric values: {examples}"
            )
        cleaned[column] = numeric_values

    return cleaned


def load_heart_disease_data(
    *,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load UCI dataset 45, caching its public tabular data locally by default.

    Set ``refresh=True`` to fetch the latest copy from UCI. Set ``cache_path=None``
    to fetch without reading or writing a local CSV cache.
    """
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists() and not refresh:
        try:
            return clean_heart_disease_data(pd.read_csv(path, na_values=["?"]))
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            raise RuntimeError(
                f"Could not read the cached dataset at '{path}'. "
                "Remove the damaged cache or call with refresh=True."
            ) from error

    try:
        repository = fetch_ucirepo(id=UCI_HEART_DISEASE_ID)
    except Exception as error:
        raise RuntimeError(
            "Could not download the UCI Heart Disease dataset. Check your internet "
            "connection, or provide a valid local cache."
        ) from error

    features = repository.data.features
    targets = repository.data.targets
    if not isinstance(features, pd.DataFrame) or not isinstance(targets, pd.DataFrame):
        raise RuntimeError("UCI returned data in an unexpected format.")
    if targets.shape[1] != 1:
        raise RuntimeError(
            f"Expected one target column from UCI; received {targets.shape[1]}."
        )
    if len(features) != len(targets):
        raise RuntimeError("UCI returned different numbers of feature and target rows.")

    target = targets.iloc[:, 0].rename(TARGET_COLUMN)
    combined = pd.concat(
        [features.reset_index(drop=True), target.reset_index(drop=True)], axis=1
    )
    cleaned = clean_heart_disease_data(combined)

    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(path, index=False)
        except OSError as error:
            raise RuntimeError(
                f"Dataset downloaded, but could not save its cache to '{path}'."
            ) from error

    return cleaned


def binarize_target(target: pd.Series) -> pd.Series:
    """Map UCI ``num`` labels to 0 (absent) or 1 (present), rejecting bad labels."""
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any():
        raise ValueError("The target contains missing or non-numeric labels.")
    if not numeric_target.isin(TARGET_ENCODING).all():
        invalid_labels = numeric_target[~numeric_target.isin(TARGET_ENCODING)]
        raise ValueError(
            "Expected UCI target labels 0, 1, 2, 3, or 4; received "
            f"{invalid_labels.unique().tolist()}."
        )

    binary_target = numeric_target.gt(0).astype("int8")
    binary_target.name = "target"
    return binary_target


def inspect_heart_disease_data(data: pd.DataFrame) -> dict[str, Any]:
    """Return a compact, data-derived summary for console or application display."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"The dataset must contain the '{TARGET_COLUMN}' target column.")

    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]
    target_counts = data[TARGET_COLUMN].value_counts(dropna=False)
    total_rows = len(data)
    distribution = {
        ("missing" if pd.isna(label) else str(label)): {
            "count": int(count),
            "percent": round(float(count) / total_rows * 100, 2),
        }
        for label, count in target_counts.items()
    }

    return {
        "rows": total_rows,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_encoding": TARGET_ENCODING,
        "missing_values": {
            column: int(count) for column, count in data.isna().sum().items()
        },
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "target_distribution": distribution,
    }


def main() -> None:
    """Download/cache and print a concise dataset inspection report."""
    data = load_heart_disease_data()
    summary = inspect_heart_disease_data(data)

    print("UCI Heart Disease dataset (ID 45)")
    print(f"Rows: {summary['rows']}")
    print(f"Features ({summary['feature_count']}): {', '.join(summary['features'])}")
    print(f"Target: {summary['target_column']} (0 = no disease; 1-4 = disease)")
    print("\nMissing values:")
    print(pd.Series(summary["missing_values"], name="missing").to_string())
    print("\nTarget distribution:")
    print(pd.DataFrame.from_dict(summary["target_distribution"], orient="index").to_string())


if __name__ == "__main__":
    main()
