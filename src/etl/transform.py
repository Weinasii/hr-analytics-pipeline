"""
transform.py
────────────
Data transformation layer — cleaning, validation, type normalization,
feature engineering. Pure functions for easy testing.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected column types after transformation
EXPECTED_DTYPES: dict[str, str] = {
    "employee_id": "object",
    "age": "int64",
    "gender": "category",
    "department": "category",
    "job_level": "category",
    "salary_eur": "float64",
    "tenure_months": "int64",
    "performance_score": "int64",
    "satisfaction_score": "float64",
    "attrition": "int64",
}


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Raise ValueError if any required column is missing.

    Args:
        df: Input DataFrame.
        required_cols: List of mandatory column names.
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed (%d columns)", len(df.columns))


def validate_no_duplicates(df: pd.DataFrame, key: str = "employee_id") -> pd.DataFrame:
    """Remove and log duplicate employee records.

    Args:
        df: Input DataFrame.
        key: Column used as the unique identifier.

    Returns:
        De-duplicated DataFrame.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=[key], keep="last")
    n_removed = n_before - len(df)
    if n_removed:
        logger.warning("Removed %d duplicate rows on '%s'", n_removed, key)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Clip salary outliers and convert to float.

    Salaries below 15,000 EUR or above 500,000 EUR are treated
    as data entry errors and replaced with the column median.

    Args:
        df: DataFrame containing a 'salary_eur' column.

    Returns:
        DataFrame with cleaned salary values.
    """
    median = df["salary_eur"].median()
    mask = (df["salary_eur"] < 15_000) | (df["salary_eur"] > 500_000)
    n_fixed = mask.sum()
    df.loc[mask, "salary_eur"] = median
    if n_fixed:
        logger.warning("Fixed %d salary outliers (replaced with median %.0f)", n_fixed, median)
    df["salary_eur"] = df["salary_eur"].astype(float)
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop missing values using context-appropriate strategies.

    - Numeric columns: median imputation
    - Categorical columns: mode imputation
    - Rows with >50% missing values: dropped

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    # Drop rows with too many nulls
    threshold = len(df.columns) * 0.5
    n_before = len(df)
    df = df.dropna(thresh=threshold)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d rows with >50%% missing values", n_dropped)

    # Impute numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Impute categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

    logger.info("Missing value imputation complete. Null count: %d", df.isnull().sum().sum())
    return df


def normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize gender labels to a consistent vocabulary.

    Args:
        df: DataFrame with a 'gender' column.

    Returns:
        DataFrame with normalized gender values.
    """
    mapping = {
        "m": "M", "male": "M", "homme": "M", "h": "M",
        "f": "F", "female": "F", "femme": "F",
        "non-binary": "Non-binary", "nonbinary": "Non-binary", "nb": "Non-binary",
    }
    df["gender"] = (
        df["gender"]
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna("Unknown")
        .astype("category")
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Segment employees into generational age bands.

    Returns:
        DataFrame with a new 'age_band' column:
        '<25' | '25-34' | '35-44' | '45-54' | '55+'
    """
    bins = [0, 25, 35, 45, 55, 200]
    labels = ["<25", "25-34", "35-44", "45-54", "55+"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df["age_band"] = df["age_band"].astype("category")
    return df


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """Segment employees by seniority tier.

    Returns:
        DataFrame with 'tenure_band': 'New (<1y)' | 'Junior (1-3y)'
        | 'Mid (3-7y)' | 'Senior (7y+)'
    """
    bins = [0, 12, 36, 84, 9999]
    labels = ["New (<1y)", "Junior (1-3y)", "Mid (3-7y)", "Senior (7y+)"]
    df["tenure_band"] = pd.cut(df["tenure_months"], bins=bins, labels=labels, right=False)
    df["tenure_band"] = df["tenure_band"].astype("category")
    return df


def add_salary_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Add each employee's salary percentile within their department.

    Returns:
        DataFrame with 'salary_percentile' (0–100).
    """
    df["salary_percentile"] = df.groupby("department")["salary_eur"].rank(pct=True).mul(100).round(1)
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple rule-based attrition risk score (0–100).

    Weighted heuristic before the ML model is applied,
    useful for interpretability with non-technical stakeholders.

    Args:
        df: DataFrame with overtime, satisfaction, tenure, and months_since_promotion.

    Returns:
        DataFrame with a 'risk_score' column.
    """
    required = ["overtime_hours_monthly", "satisfaction_score", "tenure_months", "months_since_promotion"]
    for col in required:
        if col not in df.columns:
            logger.warning("Column '%s' not found — risk_score may be incomplete", col)
            df[col] = 0

    score = (
        (df["overtime_hours_monthly"] / 40 * 30)
        + ((5 - df["satisfaction_score"]) / 4 * 30)
        + (df["months_since_promotion"].clip(0, 36) / 36 * 20)
        + ((180 - df["tenure_months"].clip(0, 180)) / 180 * 20)
    ).clip(0, 100).round(1)

    df["risk_score"] = score
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full transformation pipeline.

    Applies all cleaning, validation, and feature engineering steps
    in a deterministic order.

    Args:
        df: Raw DataFrame from the extraction layer.

    Returns:
        Transformed, analysis-ready DataFrame.
    """
    required_cols = ["employee_id", "age", "gender", "department", "salary_eur", "attrition"]
    validate_schema(df, required_cols)

    df = (
        df
        .pipe(validate_no_duplicates)
        .pipe(clean_missing_values)
        .pipe(clean_salary)
        .pipe(normalize_gender)
        .pipe(add_age_band)
        .pipe(add_tenure_band)
        .pipe(add_salary_percentile)
        .pipe(add_risk_score)
    )

    logger.info("Transformation complete — %d rows, %d columns", *df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    raw_path = Path("data/raw/employees_raw.csv")
    processed_path = Path("data/processed/employees_clean.csv")

    df_raw = pd.read_csv(raw_path)
    df_clean = run_transformation(df_raw)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    print(f"✅ Transformed data saved → {processed_path}")
    print(df_clean.dtypes)
