"""
test_transform.py
─────────────────
Unit tests for the data transformation pipeline.
Run with: pytest tests/ -v --cov=src
"""

import numpy as np
import pandas as pd
import pytest

from src.etl.transform import (
    add_age_band,
    add_risk_score,
    add_tenure_band,
    clean_missing_values,
    clean_salary,
    normalize_gender,
    run_transformation,
    validate_no_duplicates,
    validate_schema,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal valid employee DataFrame for testing."""
    return pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, 11)],
        "age": [25, 32, 45, 28, 55, 41, 33, 29, 38, 50],
        "gender": ["M", "F", "M", "F", "M", "Non-binary", "F", "M", "F", "M"],
        "department": ["Engineering"] * 5 + ["HR"] * 5,
        "job_level": ["Junior", "Mid", "Senior", "Junior", "Lead"] * 2,
        "education_level": ["Bachelor"] * 10,
        "salary_eur": [32000, 45000, 72000, 35000, 95000, 48000, 41000, 36000, 62000, 88000],
        "tenure_months": [6, 24, 84, 12, 120, 36, 18, 3, 60, 96],
        "performance_score": [3, 4, 5, 2, 4, 3, 4, 3, 5, 4],
        "satisfaction_score": [3.5, 4.2, 2.8, 3.9, 4.5, 3.0, 4.1, 2.5, 3.7, 4.8],
        "overtime_hours_monthly": [5, 10, 30, 2, 20, 8, 15, 25, 4, 12],
        "distance_km": [5, 15, 8, 30, 2, 12, 7, 20, 3, 10],
        "training_hours_ytd": [10, 20, 5, 15, 30, 8, 25, 0, 18, 40],
        "months_since_promotion": [6, 18, 36, 12, 48, 24, 8, 36, 14, 60],
        "manager_rating": [4, 5, 3, 4, 3, 4, 5, 2, 4, 3],
        "absences_ytd": [2, 5, 8, 1, 3, 10, 4, 15, 2, 6],
        "attrition": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    })


# ──────────────────────────────────────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_schema_passes_with_all_columns(self, sample_df):
        """validate_schema should not raise when all required cols are present."""
        validate_schema(sample_df, ["employee_id", "age", "gender", "salary_eur"])

    def test_schema_raises_on_missing_column(self, sample_df):
        """validate_schema should raise ValueError for missing columns."""
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(sample_df, ["employee_id", "nonexistent_column"])

    def test_no_duplicates_removes_dupes(self):
        """validate_no_duplicates should remove duplicate employee IDs."""
        df = pd.DataFrame({
            "employee_id": ["EMP001", "EMP001", "EMP002"],
            "age": [30, 30, 25],
        })
        result = validate_no_duplicates(df)
        assert len(result) == 2
        assert result["employee_id"].nunique() == 2

    def test_no_duplicates_preserves_unique_records(self, sample_df):
        """validate_no_duplicates should not drop unique records."""
        result = validate_no_duplicates(sample_df)
        assert len(result) == len(sample_df)


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCleaning:
    def test_clean_salary_replaces_outliers(self):
        """Salaries outside [15k, 500k] should be replaced with the median."""
        df = pd.DataFrame({"salary_eur": [0, 45000, 72000, 999999]})
        result = clean_salary(df)
        assert (result["salary_eur"] >= 15_000).all()
        assert (result["salary_eur"] <= 500_000).all()

    def test_clean_salary_preserves_valid_values(self, sample_df):
        """Valid salary values should not be modified."""
        original = sample_df["salary_eur"].copy()
        result = clean_salary(sample_df.copy())
        assert (result["salary_eur"] == original).all()

    def test_clean_missing_values_no_nulls_remain(self, sample_df):
        """No nulls should remain after clean_missing_values."""
        # Introduce some nulls
        df = sample_df.copy()
        df.loc[0, "satisfaction_score"] = np.nan
        df.loc[2, "department"] = None
        result = clean_missing_values(df)
        assert result.isnull().sum().sum() == 0

    def test_normalize_gender_standardizes_labels(self):
        """All gender variants should map to canonical labels."""
        df = pd.DataFrame({"gender": ["male", "FEMALE", "M", "f", "femme", "nb", "unknown_val"]})
        result = normalize_gender(df)
        valid = {"M", "F", "Non-binary", "Unknown"}
        assert set(result["gender"].unique()).issubset(valid)


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_add_age_band_covers_all_ages(self, sample_df):
        """Every employee should be assigned an age band."""
        result = add_age_band(sample_df)
        assert result["age_band"].isnull().sum() == 0

    def test_add_age_band_values(self):
        """Age band boundaries should be correct."""
        df = pd.DataFrame({"age": [20, 24, 25, 34, 35, 44, 45, 54, 55, 65]})
        result = add_age_band(df)
        assert result.loc[0, "age_band"] == "<25"
        assert result.loc[4, "age_band"] == "35-44"
        assert result.loc[8, "age_band"] == "55+"

    def test_add_tenure_band_correct_buckets(self):
        """Tenure bands should map correctly from months."""
        df = pd.DataFrame({"tenure_months": [6, 13, 40, 100]})
        result = add_tenure_band(df)
        assert result.loc[0, "tenure_band"] == "New (<1y)"
        assert result.loc[1, "tenure_band"] == "Junior (1-3y)"
        assert result.loc[2, "tenure_band"] == "Mid (3-7y)"
        assert result.loc[3, "tenure_band"] == "Senior (7y+)"

    def test_add_risk_score_range(self, sample_df):
        """Risk scores must be between 0 and 100."""
        df = sample_df.copy()
        result = add_risk_score(df)
        assert (result["risk_score"] >= 0).all()
        assert (result["risk_score"] <= 100).all()

    def test_add_risk_score_high_overtime_high_risk(self):
        """High overtime + low satisfaction should produce a high risk score."""
        df = pd.DataFrame({
            "overtime_hours_monthly": [40],
            "satisfaction_score": [1.0],
            "tenure_months": [3],
            "months_since_promotion": [36],
        })
        result = add_risk_score(df)
        assert result.loc[0, "risk_score"] > 60


# ──────────────────────────────────────────────────────────────────────────────
# Integration test
# ──────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_run_transformation_returns_expected_new_columns(self, sample_df):
        """Full pipeline should produce age_band, tenure_band, risk_score columns."""
        result = run_transformation(sample_df)
        for col in ["age_band", "tenure_band", "risk_score"]:
            assert col in result.columns, f"Expected column '{col}' not found after transformation"

    def test_run_transformation_no_nulls(self, sample_df):
        """No nulls should remain in numeric columns after full pipeline."""
        result = run_transformation(sample_df)
        numeric_nulls = result.select_dtypes(include=[np.number]).isnull().sum().sum()
        assert numeric_nulls == 0

    def test_run_transformation_row_count_stable(self, sample_df):
        """Row count should not decrease (no erroneous drops on clean data)."""
        result = run_transformation(sample_df)
        assert len(result) == len(sample_df)
