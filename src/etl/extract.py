"""
extract.py
──────────
Data extraction layer — supports CSV files, SQL databases, and REST APIs.
Designed to be easily extended with new data sources.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ──────────────────────────────────────────────────────────────────────────────
# CSV extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_from_csv(filepath: str | Path) -> pd.DataFrame:
    """Load an employee dataset from a CSV file.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        Raw DataFrame with all original columns preserved.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    logger.info("Extracting from CSV: %s", filepath)
    df = pd.read_csv(filepath, encoding="utf-8-sig", low_memory=False)

    if df.empty:
        raise ValueError(f"The file {filepath} is empty.")

    logger.info("Extracted %d rows × %d columns", *df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SQL extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_from_sql(
    query: str,
    db_url: Optional[str] = None,
) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame.

    Args:
        query: Valid SQL SELECT statement.
        db_url: SQLAlchemy connection string.
                Defaults to the DATABASE_URL environment variable.

    Returns:
        DataFrame with query results.

    Example:
        >>> df = extract_from_sql("SELECT * FROM hr.employees LIMIT 100")
    """
    db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///data/hr_sample.db")
    engine = create_engine(db_url)

    logger.info("Connecting to database…")
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    logger.info("SQL query returned %d rows × %d columns", *df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# REST API extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_from_api(
    endpoint: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch HR data from a REST API endpoint.

    Args:
        endpoint: Full URL of the API endpoint.
        headers: Optional HTTP headers (e.g. Authorization tokens).
        params: Optional query parameters.
        timeout: Request timeout in seconds.

    Returns:
        DataFrame built from the JSON response.

    Raises:
        requests.HTTPError: On non-2xx HTTP responses.
    """
    logger.info("Fetching from API: %s", endpoint)
    response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)

    logger.info("API returned %d rows × %d columns", *df.shape)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Sample data generator (for demo / testing)
# ──────────────────────────────────────────────────────────────────────────────

def generate_sample_data(n_employees: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic HR dataset.

    Args:
        n_employees: Number of employee records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame ready for the transformation pipeline.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    departments = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"]
    job_levels = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]
    education = ["High School", "Bachelor", "Master", "PhD"]

    n = n_employees
    df = pd.DataFrame({
        "employee_id": [f"EMP{str(i).zfill(5)}" for i in range(1, n + 1)],
        "age": rng.integers(22, 62, n),
        "gender": rng.choice(["M", "F", "Non-binary"], n, p=[0.48, 0.48, 0.04]),
        "department": rng.choice(departments, n),
        "job_level": rng.choice(job_levels, n, p=[0.20, 0.30, 0.25, 0.12, 0.10, 0.03]),
        "education_level": rng.choice(education, n, p=[0.10, 0.45, 0.35, 0.10]),
        "salary_eur": rng.integers(28_000, 120_000, n),
        "tenure_months": rng.integers(1, 180, n),
        "performance_score": rng.integers(1, 6, n),
        "satisfaction_score": rng.uniform(1, 5, n).round(1),
        "overtime_hours_monthly": rng.integers(0, 40, n),
        "distance_km": rng.integers(1, 80, n),
        "training_hours_ytd": rng.integers(0, 60, n),
        "months_since_promotion": rng.integers(0, 60, n),
        "manager_rating": rng.integers(1, 6, n),
        "absences_ytd": rng.integers(0, 30, n),
        "attrition": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "hire_date": pd.to_datetime("2015-01-01") + pd.to_timedelta(
            rng.integers(0, 365 * 8, n), unit="D"
        ),
    })

    output_path = RAW_DIR / "employees_raw.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Sample data saved to %s (%d employees)", output_path, n)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HR Analytics — Data Extraction")
    parser.add_argument(
        "--source",
        choices=["csv", "sql", "api", "sample"],
        default="sample",
        help="Data source type",
    )
    parser.add_argument("--path", type=str, help="Path or URL for csv/api sources")
    parser.add_argument("--n", type=int, default=500, help="Rows for sample generation")
    args = parser.parse_args()

    if args.source == "sample":
        df = generate_sample_data(n_employees=args.n)
    elif args.source == "csv":
        df = extract_from_csv(args.path)
    elif args.source == "sql":
        df = extract_from_sql(args.path)
    elif args.source == "api":
        df = extract_from_api(args.path)

    print(f"\n✅ Extraction complete — {len(df)} rows loaded.\n")
    print(df.head(3).to_string())
