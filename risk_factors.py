"""
EduAI – Transparent Risk Factor Identification
================================================
Steps covered: 7 (rule-based risk factors), 8 (individual analysis helper)

This module is SEPARATE from the Machine Learning model.

  ML model  → predicts overall academic risk level
  This module → identifies observable academic warning indicators

Do NOT interpret risk factors as proven causes of academic performance.
They are transparent, rule-based observations presented to educators.
"""

from typing import List, Dict

# ── Thresholds (clearly documented) ──────────────────────────────────────────
THRESHOLD_LOW_MARKS              = 50.0   # below this → "Low marks"
THRESHOLD_LOW_ATTENDANCE         = 65.0   # below this → "Low attendance"
THRESHOLD_LOW_QUIZ               = 50.0   # below this → "Low quiz performance"
THRESHOLD_LOW_ASSIGNMENT         = 60.0   # below this → "Low assignment completion"
THRESHOLD_DECLINING_PERFORMANCE  = -8.0   # marks - previous_marks < this


def identify_risk_factors(marks: float,
                          attendance: float,
                          quiz_score: float,
                          assignment_completion: float,
                          previous_marks: float) -> List[str]:
    """
    Identify observable academic risk indicators for a student.

    Rules (transparent & documented)
    ---------------------------------
    1. Low marks              : marks < 50
    2. Low attendance         : attendance < 65 %
    3. Low quiz performance   : quiz_score < 50
    4. Low assignment completion: assignment_completion < 60 %
    5. Declining performance  : (marks - previous_marks) < -8

    Parameters
    ----------
    marks, attendance, quiz_score, assignment_completion, previous_marks
        Float values representing current academic indicators.

    Returns
    -------
    List[str]  – list of human-readable risk factor strings
                 (empty list if no factors detected)
    """
    factors: List[str] = []

    if marks < THRESHOLD_LOW_MARKS:
        factors.append("Low marks")

    if attendance < THRESHOLD_LOW_ATTENDANCE:
        factors.append("Low attendance")

    if quiz_score < THRESHOLD_LOW_QUIZ:
        factors.append("Low quiz performance")

    if assignment_completion < THRESHOLD_LOW_ASSIGNMENT:
        factors.append("Low assignment completion")

    delta = marks - previous_marks
    if delta < THRESHOLD_DECLINING_PERFORMANCE:
        factors.append("Declining academic performance")

    return factors


def build_risk_factor_summary(student_id: str,
                              risk_level: str,
                              factors: List[str]) -> str:
    """
    Format a human-readable risk factor summary for display.

    Parameters
    ----------
    student_id : str
    risk_level : str   – ML-predicted label
    factors    : List[str]

    Returns
    -------
    str  – formatted text block
    """
    lines = [
        f"Student ID: {student_id}",
        f"Predicted Risk Level: {risk_level}",
        "",
        "Observable Academic Indicators:",
    ]
    if factors:
        for f in factors:
            lines.append(f"  • {f}")
    else:
        lines.append("  • No significant academic risk indicators detected.")

    lines.append("")
    lines.append("Note: These indicators are based on transparent rules and are")
    lines.append("intended to support, not replace, educator judgment.")
    return "\n".join(lines)


def get_risk_factor_frequency(df_results) -> Dict[str, int]:
    """
    Count how many students are affected by each risk factor.

    Parameters
    ----------
    df_results : pd.DataFrame  – must contain a 'risk_factors' column
                                 where each cell is a list of factor strings.

    Returns
    -------
    dict  – {factor_label: count}
    """
    from collections import Counter
    counter: Counter = Counter()
    for factors in df_results["risk_factors"]:
        if isinstance(factors, list):
            counter.update(factors)
    return dict(counter)
