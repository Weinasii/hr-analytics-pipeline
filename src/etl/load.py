"""
load.py
───────
Data loading layer — writes transformed data to target destinations
(CSV files, PostgreSQL, or data warehouses).
"""
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to CSV. Creates parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)

def load_to_sql(df: pd.DataFrame, table: str, db_url: str, if_exists: str = "replace") -> None:
    """Write DataFrame to a SQL table using SQLAlchemy."""
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    df.to_sql(table, engine, if_exists=if_exists, index=False, schema="hr")
    logger.info("Loaded %d rows → hr.%s (%s)", len(df), table, if_exists)
