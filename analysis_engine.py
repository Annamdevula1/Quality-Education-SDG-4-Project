"""
EduAI – Student Analysis Engine
==================================
Steps covered: 8 (individual analysis), 15 (final results table)

Provides the complete analysis pipeline for a single student and for
the full dataset:
  1. ML risk prediction
  2. Transparent risk factor identification
  3. IBM Granite explanation + recommendations + monitoring suggestion
  4. Structured result record
"""

import pandas as pd
from typing import Dict, List, Optional

from data_processing import FEATURE_COLS
from ml_model import predict_single, load_model
from risk_factors import identify_risk_factors
from granite_integration import generate_granite_response


# ── Column order for the final results table ──────────────────────────────────
RESULT_COLUMNS = [
    "student_id",
    "marks",
    "attendance",
    "quiz_score",
    "assignment_completion",
    "previous_marks",
    "predicted_risk_level",
    "risk_factors",
    "granite_explanation",
    "recommendation_1",
    "recommendation_2",
    "recommendation_3",
    "monitoring_suggestion",
    "ai_source",
]


def analyse_student(student_id: str,
                    marks: float,
                    attendance: float,
                    quiz_score: float,
                    assignment_completion: float,
                    previous_marks: float,
                    model=None,
                    le=None) -> Dict:
    """
    Run the complete EduAI analysis pipeline for a single student.

    Parameters
    ----------
    student_id               : str
    marks                    : float  (0–100)
    attendance               : float  (0–100 %)
    quiz_score               : float  (0–100)
    assignment_completion    : float  (0–100 %)
    previous_marks           : float  (0–100)
    model                    : fitted RandomForestClassifier (optional – loaded from disk if None)
    le                       : fitted LabelEncoder          (optional – loaded from disk if None)

    Returns
    -------
    dict  – full analysis result (matches RESULT_COLUMNS)
    """
    # ── Load model if not provided ─────────────────────────────────────────
    if model is None or le is None:
        model, le = load_model()

    features = {
        "marks":                 marks,
        "attendance":            attendance,
        "quiz_score":            quiz_score,
        "assignment_completion": assignment_completion,
        "previous_marks":        previous_marks,
    }

    # ── Step 1: ML Risk Prediction ─────────────────────────────────────────
    risk_level = predict_single(model, le, features)

    # ── Step 2: Rule-based Risk Factors ───────────────────────────────────
    factors = identify_risk_factors(
        marks=marks,
        attendance=attendance,
        quiz_score=quiz_score,
        assignment_completion=assignment_completion,
        previous_marks=previous_marks,
    )

    # ── Step 3: IBM Granite (or DEMO fallback) ─────────────────────────────
    granite = generate_granite_response(
        student_id=student_id,
        marks=marks,
        attendance=attendance,
        quiz_score=quiz_score,
        assignment_completion=assignment_completion,
        previous_marks=previous_marks,
        risk_level=risk_level,
        risk_factors=factors,
    )

    # Pad recommendations to always have exactly 3 entries
    recs = granite.get("recommendations", [])
    while len(recs) < 3:
        recs.append("(Recommendation not available)")

    ai_source = "DEMO – Not IBM Granite" if granite.get("is_demo") else "IBM Granite"

    # ── Step 4: Assemble result ────────────────────────────────────────────
    return {
        "student_id":            student_id,
        "marks":                 marks,
        "attendance":            attendance,
        "quiz_score":            quiz_score,
        "assignment_completion": assignment_completion,
        "previous_marks":        previous_marks,
        "predicted_risk_level":  risk_level,
        "risk_factors":          factors,
        "granite_explanation":   granite.get("explanation", ""),
        "recommendation_1":      recs[0],
        "recommendation_2":      recs[1],
        "recommendation_3":      recs[2],
        "monitoring_suggestion": granite.get("monitoring_suggestion", ""),
        "ai_source":             ai_source,
        "parse_success":         granite.get("parse_success", False),
        "parse_errors":          granite.get("parse_errors", []),
    }


def analyse_dataframe(df: pd.DataFrame,
                      model=None,
                      le=None,
                      progress_callback=None) -> pd.DataFrame:
    """
    Run analyse_student() for every row in df and return a results DataFrame.

    Parameters
    ----------
    df               : pd.DataFrame  – must contain student_id + FEATURE_COLS
    model, le        : optional pre-loaded ML model and encoder
    progress_callback: optional callable(current, total) for UI progress bars

    Returns
    -------
    pd.DataFrame  with RESULT_COLUMNS
    """
    if model is None or le is None:
        model, le = load_model()

    records = []
    total = len(df)

    for i, row in df.iterrows():
        result = analyse_student(
            student_id=row["student_id"],
            marks=float(row["marks"]),
            attendance=float(row["attendance"]),
            quiz_score=float(row["quiz_score"]),
            assignment_completion=float(row["assignment_completion"]),
            previous_marks=float(row["previous_marks"]),
            model=model,
            le=le,
        )
        records.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    results_df = pd.DataFrame(records)
    return results_df


def filter_results(df: pd.DataFrame, risk_level: Optional[str] = None) -> pd.DataFrame:
    """
    Filter the results DataFrame by predicted risk level.

    Parameters
    ----------
    df         : results DataFrame (from analyse_dataframe)
    risk_level : "On Track" | "Needs Attention" | "At Risk" | None (all)

    Returns
    -------
    Filtered pd.DataFrame
    """
    if risk_level is None:
        return df
    return df[df["predicted_risk_level"] == risk_level].reset_index(drop=True)
