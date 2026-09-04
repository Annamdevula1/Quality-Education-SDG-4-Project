"""
EduAI – Synthetic Student Data Generator
=========================================
Steps covered: 1, 2 (data creation), 3 (risk-label logic)

Creates a reproducible, anonymized student dataset with NO personally
identifiable information.  All data is entirely synthetic.

Risk-label logic (transparent & documented):
  At Risk        : (marks < 45)
                   OR (attendance < 55)
                   OR (quiz_score < 40 AND assignment_completion < 50)
  Needs Attention: not At Risk AND (
                      marks < 60
                   OR attendance < 70
                   OR quiz_score < 55
                   OR assignment_completion < 60
                   OR (marks - previous_marks) < -10
                   )
  On Track       : all other students
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 500          # sufficient for meaningful ML


def _generate_segment(rng, n, marks_range, attend_range,
                      quiz_range, assign_range, prev_marks_range):
    """Generate one homogeneous performance segment."""
    marks                 = rng.uniform(*marks_range,  n)
    attendance            = rng.uniform(*attend_range,  n)
    quiz_score            = rng.uniform(*quiz_range,    n)
    assignment_completion = rng.uniform(*assign_range,  n)
    previous_marks        = rng.uniform(*prev_marks_range, n)
    return marks, attendance, quiz_score, assignment_completion, previous_marks


def generate_student_data(n_students: int = N_STUDENTS,
                          seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic, reproducible student dataset.

    Parameters
    ----------
    n_students : int
        Total number of student records (default 500).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: student_id, marks, attendance, quiz_score,
                 assignment_completion, previous_marks, risk_level
    """
    rng = np.random.default_rng(seed)

    # ── Segment sizes (proportional representation of all three classes) ──
    n_at_risk   = int(n_students * 0.25)   # ~125
    n_attention = int(n_students * 0.35)   # ~175
    n_on_track  = n_students - n_at_risk - n_attention  # ~200

    # ── At Risk segment ───────────────────────────────────────────────────
    ar_marks, ar_attend, ar_quiz, ar_assign, ar_prev = _generate_segment(
        rng, n_at_risk,
        marks_range=(18, 52),
        attend_range=(28, 63),
        quiz_range=(12, 50),
        assign_range=(18, 58),
        prev_marks_range=(22, 62),
    )

    # ── Needs Attention segment ───────────────────────────────────────────
    na_marks, na_attend, na_quiz, na_assign, na_prev = _generate_segment(
        rng, n_attention,
        marks_range=(44, 66),
        attend_range=(60, 82),
        quiz_range=(44, 66),
        assign_range=(54, 76),
        prev_marks_range=(48, 76),
    )
    # Introduce declining academic performance in ~40% of this segment
    decline_idx = rng.choice(n_attention, size=int(n_attention * 0.40),
                             replace=False)
    na_prev[decline_idx] = na_marks[decline_idx] + rng.uniform(11, 26,
                                                                len(decline_idx))

    # ── On Track segment ──────────────────────────────────────────────────
    ot_marks, ot_attend, ot_quiz, ot_assign, ot_prev = _generate_segment(
        rng, n_on_track,
        marks_range=(60, 100),
        attend_range=(75, 100),
        quiz_range=(60, 100),
        assign_range=(70, 100),
        prev_marks_range=(58, 100),
    )
    # Natural small variation (some slight declines are realistic)
    ot_prev = ot_marks + rng.uniform(-8, 13, n_on_track)

    # ── Concatenate all segments ──────────────────────────────────────────
    all_marks  = np.concatenate([ar_marks,  na_marks,  ot_marks])
    all_attend = np.concatenate([ar_attend, na_attend, ot_attend])
    all_quiz   = np.concatenate([ar_quiz,   na_quiz,   ot_quiz])
    all_assign = np.concatenate([ar_assign, na_assign, ot_assign])
    all_prev   = np.concatenate([ar_prev,   na_prev,   ot_prev])
    labels     = (
        ["At Risk"]         * n_at_risk
        + ["Needs Attention"] * n_attention
        + ["On Track"]       * n_on_track
    )

    # ── Clip to valid academic ranges ─────────────────────────────────────
    all_marks  = np.clip(all_marks,  0, 100)
    all_attend = np.clip(all_attend, 0, 100)
    all_quiz   = np.clip(all_quiz,   0, 100)
    all_assign = np.clip(all_assign, 0, 100)
    all_prev   = np.clip(all_prev,   0, 100)

    # ── Shuffle and assign IDs ────────────────────────────────────────────
    idx = rng.permutation(n_students)
    student_ids = [f"STU{str(i + 1).zfill(3)}" for i in range(n_students)]

    df = pd.DataFrame({
        "student_id":            [student_ids[i] for i in idx],
        "marks":                 np.round(all_marks[idx], 1),
        "attendance":            np.round(all_attend[idx], 1),
        "quiz_score":            np.round(all_quiz[idx], 1),
        "assignment_completion": np.round(all_assign[idx], 1),
        "previous_marks":        np.round(all_prev[idx], 1),
        "risk_level":            [labels[i] for i in idx],
    })

    return df


def apply_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Re)apply the transparent rule-based risk labels to any DataFrame
    that contains the five academic feature columns.

    Returns a copy of df with 'risk_level' added or replaced.
    """
    def _label(row):
        if (row["marks"] < 45
                or row["attendance"] < 55
                or (row["quiz_score"] < 40
                    and row["assignment_completion"] < 50)):
            return "At Risk"
        delta = row["marks"] - row["previous_marks"]
        if (row["marks"] < 60
                or row["attendance"] < 70
                or row["quiz_score"] < 55
                or row["assignment_completion"] < 60
                or delta < -10):
            return "Needs Attention"
        return "On Track"

    df = df.copy()
    df["risk_level"] = df.apply(_label, axis=1)
    return df
