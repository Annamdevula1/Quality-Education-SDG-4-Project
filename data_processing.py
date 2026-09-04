"""
EduAI – Data Validation & Processing Pipeline
==============================================
Steps covered: 2 (full processing pipeline)

Provides:
  - load_and_validate()  : load, inspect, validate, and clean the dataset
  - prepare_features()   : extract feature matrix X and label vector y
"""

import pandas as pd
import numpy as np

# ── Valid academic ranges ─────────────────────────────────────────────────────
RANGE_CHECKS = {
    "marks":                 (0, 100),
    "attendance":            (0, 100),
    "quiz_score":            (0, 100),
    "assignment_completion": (0, 100),
    "previous_marks":        (0, 100),
}

FEATURE_COLS = [
    "marks",
    "attendance",
    "quiz_score",
    "assignment_completion",
    "previous_marks",
]

TARGET_COL = "risk_level"

RISK_ORDER = ["On Track", "Needs Attention", "At Risk"]


# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(df: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Validate, inspect, and clean the student DataFrame.

    Steps performed
    ---------------
    1. Display sample rows
    2. Report dataset shape
    3. Check data types
    4. Check missing values  → fill numeric NaN with column median
    5. Check duplicate records  → drop exact duplicates
    6. Descriptive statistics
    7. Validate data ranges
    8. Return clean DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        Raw student data (from generate_student_data or a CSV load).
    verbose : bool
        Print the report to stdout.

    Returns
    -------
    pd.DataFrame  (clean copy)
    """
    lines = []

    lines.append("=" * 60)
    lines.append("  EduAI – Data Validation & Processing Report")
    lines.append("=" * 60)

    # 1. Sample rows
    lines.append("\n[1] Sample Records (first 5):")
    lines.append(df.head().to_string(index=False))

    # 2. Shape
    lines.append(f"\n[2] Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # 3. Data types
    lines.append("\n[3] Data Types:")
    for col, dtype in df.dtypes.items():
        lines.append(f"    {col:<26} {dtype}")

    # 4. Missing values
    missing = df.isnull().sum()
    lines.append("\n[4] Missing Values:")
    if missing.sum() == 0:
        lines.append("    No missing values detected.")
    else:
        for col, cnt in missing[missing > 0].items():
            lines.append(f"    {col}: {cnt} missing  → filled with column median")
        # Fill numeric columns with their median
        for col in FEATURE_COLS:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # 5. Duplicates
    dup_count = df.duplicated().sum()
    lines.append(f"\n[5] Duplicate Records: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        lines.append(f"    → {dup_count} duplicate(s) dropped. New shape: {df.shape}")

    # 6. Descriptive statistics
    lines.append("\n[6] Descriptive Statistics:")
    lines.append(df[FEATURE_COLS].describe().round(2).to_string())

    # 7. Range validation
    lines.append("\n[7] Data Range Validation:")
    all_valid = True
    for col, (lo, hi) in RANGE_CHECKS.items():
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        status = "OK" if out_of_range == 0 else f"WARNING – {out_of_range} out-of-range"
        lines.append(f"    {col:<26} [{lo}, {hi}]  → {status}")
        if out_of_range > 0:
            all_valid = False
            df[col] = df[col].clip(lo, hi)

    if not all_valid:
        lines.append("    Out-of-range values have been clipped to valid bounds.")

    # 8. Risk-level distribution
    if TARGET_COL in df.columns:
        lines.append("\n[8] Risk Level Distribution:")
        counts = df[TARGET_COL].value_counts()
        for level in RISK_ORDER:
            n = counts.get(level, 0)
            pct = 100 * n / len(df)
            lines.append(f"    {level:<18} {n:>4}  ({pct:.1f}%)")

    lines.append("\n[✓] Data validation complete.\n")
    lines.append("=" * 60)

    report = "\n".join(lines)
    if verbose:
        print(report)

    df.attrs["validation_report"] = report
    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame):
    """
    Extract the feature matrix X and label vector y from the clean DataFrame.

    Returns
    -------
    X : pd.DataFrame  – five academic feature columns
    y : pd.Series     – risk_level labels
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y
