"""Data loading, cleaning and splitting for the Annapurna project."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAT_COLS = ["day_of_week", "meal", "menu_main"]
FLAG_COLS = ["exam_week", "holiday_or_event", "rain"]
NUM_COLS = ["temperature_c", "students_registered"]

# Inputs known BEFORE a meal is cooked (used for the regression model)
REG_FEATURES = CAT_COLS + FLAG_COLS + NUM_COLS
REG_TARGET = "plates_eaten"
CLF_TARGET = "high_waste"


def find_data_path():
    """Find the Annapurna CSV inside the project (data/raw, data/processed, ...)."""
    search_roots = [PROJECT_ROOT / "data", PROJECT_ROOT]
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if ".venv" in str(path):
                continue
            try:
                cols = pd.read_csv(path, nrows=0).columns
            except Exception:
                continue
            if REG_TARGET in cols and CLF_TARGET in cols:
                return path
    raise FileNotFoundError(
        "Could not find the Annapurna CSV. Put it in data/raw/ and try again."
    )


def load_raw(path=None):
    """Load the raw dataset."""
    return pd.read_csv(path or find_data_path())


def clean_data(df):
    """Fill missing temperatures with the median; drop rows missing the target."""
    df = df.copy()
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].median())
    df = df.dropna(subset=[REG_TARGET])
    return df


def load_clean_data(path=None):
    """Load and clean in one step."""
    return clean_data(load_raw(path))


def iqr_limits(series):
    """Return the lower and upper IQR limits for a numeric column."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def find_outliers(df, col):
    """Rows whose value in `col` is outside the IQR limits."""
    lower, upper = iqr_limits(df[col])
    return df[(df[col] < lower) | (df[col] > upper)], lower, upper


def get_regression_data(df):
    """Inputs (X) and target (y) for predicting plates eaten."""
    return df[REG_FEATURES], df[REG_TARGET]


def split_data(X, y, seed=42):
    """70% train / 15% validation / 15% test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
