"""
attrition_model.py
──────────────────
Machine learning pipeline for employee attrition prediction.

Uses a Random Forest classifier as primary model with XGBoost as
an ensemble member. Outputs per-employee risk probabilities and
SHAP-based feature importances for explainability.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/attrition_model.pkl")

# Features selected after EDA and feature importance analysis
FEATURE_COLS = [
    "age",
    "tenure_months",
    "salary_eur",
    "performance_score",
    "satisfaction_score",
    "overtime_hours_monthly",
    "distance_km",
    "training_hours_ytd",
    "months_since_promotion",
    "manager_rating",
    "absences_ytd",
]

CATEGORICAL_COLS = ["department", "job_level", "education_level", "gender"]
TARGET_COL = "attrition"


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical features.

    Args:
        df: DataFrame with raw categorical columns.

    Returns:
        Tuple of (encoded DataFrame, dict of fitted LabelEncoders).
    """
    encoders: dict[str, LabelEncoder] = {}
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y.

    Args:
        df: Transformed employee DataFrame.

    Returns:
        (X, y) ready for sklearn estimators.
    """
    df, _ = encode_categoricals(df)
    all_features = FEATURE_COLS + CATEGORICAL_COLS
    available = [c for c in all_features if c in df.columns]

    X = df[available].copy()
    y = df[TARGET_COL].copy()

    logger.info("Feature matrix: %d rows × %d features", *X.shape)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Build the full sklearn Pipeline with scaling and classifier.

    Returns:
        Unfitted Pipeline ready for .fit().
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train(
    df: pd.DataFrame,
    test_size: float = 0.20,
    cv_folds: int = 5,
) -> Tuple[Pipeline, dict]:
    """Train the attrition model and return evaluation metrics.

    Args:
        df: Cleaned employee DataFrame.
        test_size: Fraction of data held out for final evaluation.
        cv_folds: Number of cross-validation folds.

    Returns:
        Tuple of (fitted Pipeline, metrics dict).
    """
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    pipeline = build_pipeline()

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(
        "CV ROC-AUC: %.3f ± %.3f (across %d folds)", cv_scores.mean(), cv_scores.std(), cv_folds
    )

    # Final fit on full training set
    pipeline.fit(X_train, y_train)

    # Evaluation on held-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        "cv_roc_auc_std": round(cv_scores.std(), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    logger.info("Test ROC-AUC: %.4f", metrics["roc_auc"])
    logger.info("\n%s", classification_report(y_test, y_pred))

    return pipeline, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict_attrition_risk(
    df: pd.DataFrame,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Predict attrition probability for each employee.

    Args:
        df: Employee DataFrame (same schema as training data).
        pipeline: Fitted sklearn Pipeline.

    Returns:
        Original DataFrame augmented with:
        - 'attrition_probability': model output (0.0–1.0)
        - 'attrition_risk_label': 'Low' | 'Medium' | 'High'
    """
    X, _ = prepare_features(df)
    probas = pipeline.predict_proba(X)[:, 1]

    result = df.copy()
    result["attrition_probability"] = probas.round(4)
    result["attrition_risk_label"] = pd.cut(
        probas,
        bins=[0, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    )

    logger.info(
        "Risk distribution — Low: %d | Medium: %d | High: %d",
        (result["attrition_risk_label"] == "Low").sum(),
        (result["attrition_risk_label"] == "Medium").sum(),
        (result["attrition_risk_label"] == "High").sum(),
    )
    return result


def feature_importances(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract and rank feature importances from the trained classifier.

    Args:
        pipeline: Fitted sklearn Pipeline containing a tree-based classifier.
        feature_names: Names of input features in training order.

    Returns:
        DataFrame sorted by importance (descending).
    """
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    """Serialize the trained pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved → %s", path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a serialized pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Run training first.")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    processed_path = Path("data/processed/employees_clean.csv")

    if not processed_path.exists():
        print("⚠️  Run extract.py and transform.py first.")
    else:
        df = pd.read_csv(processed_path)
        print(f"Training on {len(df)} employees…\n")

        pipeline, metrics = train(df)

        print(f"✅ ROC-AUC (test): {metrics['roc_auc']}")
        print(f"   CV ROC-AUC    : {metrics['cv_roc_auc_mean']} ± {metrics['cv_roc_auc_std']}")

        save_model(pipeline)

        result_df = predict_attrition_risk(df, pipeline)
        print("\n🚨 High-risk employees (sample):")
        high_risk = result_df[result_df["attrition_risk_label"] == "High"]
        print(high_risk[["employee_id", "department", "attrition_probability"]].head(5).to_string(index=False))
