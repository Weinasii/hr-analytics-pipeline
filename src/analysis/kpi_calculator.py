"""
kpi_calculator.py
─────────────────
Core HR KPI computation engine.

All functions accept a cleaned DataFrame and return a scalar, Series,
or DataFrame — no side effects, fully testable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# KPI Result Container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HRKPIReport:
    """Structured container for all computed HR KPIs."""

    # Workforce snapshot
    headcount: int
    avg_age: float
    avg_tenure_months: float
    gender_ratio_pct_female: float

    # Attrition
    attrition_rate_pct: float
    attrition_by_dept: dict

    # Compensation
    avg_salary_eur: float
    median_salary_eur: float
    gender_pay_gap_pct: float
    salary_spread_coeff: float  # P90 / P10 ratio

    # Performance & engagement
    avg_performance_score: float
    avg_satisfaction_score: float
    employee_nps: float  # derived from satisfaction

    # Wellbeing
    absenteeism_rate_pct: float
    avg_overtime_monthly: float
    burnout_risk_pct: float  # % with risk_score > 70

    # Development
    avg_training_hours: float
    promotion_rate_pct: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"📊 HR KPI Report\n"
            f"{'─' * 40}\n"
            f"  Headcount         : {self.headcount}\n"
            f"  Attrition rate    : {self.attrition_rate_pct:.1f}%\n"
            f"  Avg salary        : €{self.avg_salary_eur:,.0f}\n"
            f"  Gender pay gap    : {self.gender_pay_gap_pct:.1f}%\n"
            f"  Satisfaction score: {self.avg_satisfaction_score:.2f} / 5\n"
            f"  Absenteeism rate  : {self.absenteeism_rate_pct:.1f}%\n"
            f"  Burnout risk      : {self.burnout_risk_pct:.1f}%\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Individual KPI functions
# ──────────────────────────────────────────────────────────────────────────────

def attrition_rate(df: pd.DataFrame) -> float:
    """Compute the overall voluntary attrition rate.

    Formula: (employees who left / total headcount) × 100

    Args:
        df: Cleaned employee DataFrame with 'attrition' column (0/1).

    Returns:
        Attrition rate as a percentage (0–100).
    """
    if "attrition" not in df.columns:
        raise KeyError("Column 'attrition' is required.")
    return round(df["attrition"].mean() * 100, 2)


def attrition_by_department(df: pd.DataFrame) -> pd.Series:
    """Compute attrition rates broken down by department.

    Returns:
        Series indexed by department, values are attrition percentages.
    """
    return (
        df.groupby("department")["attrition"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )


def gender_pay_gap(df: pd.DataFrame) -> float:
    """Compute the unadjusted gender pay gap (M vs F).

    Formula: ((avg_salary_M - avg_salary_F) / avg_salary_M) × 100

    Returns:
        Pay gap as a percentage. Positive = women earn less.
    """
    genders = df.groupby("gender")["salary_eur"].mean()
    if "M" not in genders or "F" not in genders:
        return 0.0
    gap = (genders["M"] - genders["F"]) / genders["M"] * 100
    return round(gap, 2)


def absenteeism_rate(df: pd.DataFrame, working_days_per_year: int = 220) -> float:
    """Compute the absenteeism rate as a percentage of expected working days.

    Args:
        df: DataFrame with 'absences_ytd' column.
        working_days_per_year: Denominator for rate calculation.

    Returns:
        Absenteeism rate (%).
    """
    if "absences_ytd" not in df.columns:
        return 0.0
    return round(df["absences_ytd"].mean() / working_days_per_year * 100, 2)


def employee_nps(df: pd.DataFrame) -> float:
    """Derive an Employee Net Promoter Score from satisfaction scores.

    Maps satisfaction (1–5) to NPS buckets:
        - Promoters  : score ≥ 4.5 → +1
        - Passives   : 3.5 ≤ score < 4.5 → 0
        - Detractors : score < 3.5 → -1

    Returns:
        eNPS in the range [-100, +100].
    """
    scores = df["satisfaction_score"]
    promoters = (scores >= 4.5).mean()
    detractors = (scores < 3.5).mean()
    return round((promoters - detractors) * 100, 1)


def salary_spread(df: pd.DataFrame) -> float:
    """Compute the P90/P10 salary ratio as an inequality indicator.

    Returns:
        Ratio ≥ 1. Values above 4 may signal compensation inequity.
    """
    p10 = df["salary_eur"].quantile(0.10)
    p90 = df["salary_eur"].quantile(0.90)
    return round(p90 / p10, 2) if p10 > 0 else 0.0


def burnout_risk_percentage(df: pd.DataFrame, threshold: float = 70.0) -> float:
    """Return the share of employees above the burnout risk threshold.

    Args:
        df: DataFrame with 'risk_score' column (0–100).
        threshold: Score above which an employee is considered at risk.

    Returns:
        Percentage of at-risk employees.
    """
    if "risk_score" not in df.columns:
        return 0.0
    return round((df["risk_score"] > threshold).mean() * 100, 1)


def headcount_evolution(df: pd.DataFrame, freq: str = "M") -> pd.Series:
    """Compute monthly headcount based on hire dates.

    Args:
        df: DataFrame with 'hire_date' column.
        freq: Resampling frequency ('M' = monthly, 'Q' = quarterly).

    Returns:
        Series with cumulative headcount over time.
    """
    if "hire_date" not in df.columns:
        raise KeyError("Column 'hire_date' required for evolution.")
    hires = df.set_index("hire_date").resample(freq).size().cumsum()
    hires.name = "headcount"
    return hires


def top_attrition_risk_employees(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the n employees with the highest attrition risk score.

    Useful for HR business partner targeting.

    Args:
        df: DataFrame with 'risk_score' and employee metadata.
        top_n: Number of high-risk profiles to return.

    Returns:
        Sorted DataFrame with top_n high-risk employees.
    """
    cols = [c for c in [
        "employee_id", "department", "job_level", "tenure_months",
        "satisfaction_score", "overtime_hours_monthly", "risk_score"
    ] if c in df.columns]
    return (
        df[cols]
        .sort_values("risk_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full report
# ──────────────────────────────────────────────────────────────────────────────

def compute_full_report(df: pd.DataFrame) -> HRKPIReport:
    """Compute all HR KPIs from a cleaned DataFrame.

    Args:
        df: Transformed employee DataFrame.

    Returns:
        HRKPIReport dataclass with all metrics populated.
    """
    return HRKPIReport(
        headcount=len(df),
        avg_age=round(df["age"].mean(), 1),
        avg_tenure_months=round(df["tenure_months"].mean(), 1),
        gender_ratio_pct_female=round((df["gender"] == "F").mean() * 100, 1),

        attrition_rate_pct=attrition_rate(df),
        attrition_by_dept=attrition_by_department(df).to_dict(),

        avg_salary_eur=round(df["salary_eur"].mean(), 0),
        median_salary_eur=round(df["salary_eur"].median(), 0),
        gender_pay_gap_pct=gender_pay_gap(df),
        salary_spread_coeff=salary_spread(df),

        avg_performance_score=round(df["performance_score"].mean(), 2),
        avg_satisfaction_score=round(df["satisfaction_score"].mean(), 2),
        employee_nps=employee_nps(df),

        absenteeism_rate_pct=absenteeism_rate(df),
        avg_overtime_monthly=round(df["overtime_hours_monthly"].mean(), 1),
        burnout_risk_pct=burnout_risk_percentage(df),

        avg_training_hours=round(df.get("training_hours_ytd", pd.Series([0])).mean(), 1),
        promotion_rate_pct=round(
            (df.get("months_since_promotion", pd.Series([99])) <= 12).mean() * 100, 1
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    processed_path = Path("data/processed/employees_clean.csv")
    if not processed_path.exists():
        print("⚠️  Processed data not found. Run transform.py first.")
    else:
        df = pd.read_csv(processed_path)
        report = compute_full_report(df)
        print(report.summary())

        top_risk = top_attrition_risk_employees(df, top_n=5)
        print("\n🚨 Top 5 employees at attrition risk:\n")
        print(top_risk.to_string(index=False))
