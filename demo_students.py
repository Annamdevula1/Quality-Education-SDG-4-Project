"""
EduAI – Demonstration Students
================================
Step 9: Predefined synthetic students representing all three risk categories.

ALL DATA IS SYNTHETIC.  No real student information is used.
"""

from typing import List, Dict

# ── Demonstration student definitions ─────────────────────────────────────────
# Each record contains the five academic features plus a display label.
# These are used to demonstrate the complete EduAI workflow.

DEMO_STUDENTS: List[Dict] = [
    # ── On Track ──────────────────────────────────────────────────────────
    {
        "student_id":            "DEMO-001",
        "marks":                 82.0,
        "attendance":            91.0,
        "quiz_score":            78.0,
        "assignment_completion": 88.0,
        "previous_marks":        79.0,
        "_demo_label":           "On Track",
        "_description":          "High achiever with strong attendance and consistent performance.",
    },
    {
        "student_id":            "DEMO-002",
        "marks":                 74.0,
        "attendance":            85.0,
        "quiz_score":            70.0,
        "assignment_completion": 80.0,
        "previous_marks":        72.0,
        "_demo_label":           "On Track",
        "_description":          "Steady student with above-average marks and good attendance.",
    },
    # ── Needs Attention ───────────────────────────────────────────────────
    {
        "student_id":            "DEMO-003",
        "marks":                 55.0,
        "attendance":            68.0,
        "quiz_score":            52.0,
        "assignment_completion": 62.0,
        "previous_marks":        68.0,
        "_demo_label":           "Needs Attention",
        "_description":          "Declining marks compared to previous period; borderline attendance.",
    },
    {
        "student_id":            "DEMO-004",
        "marks":                 58.0,
        "attendance":            72.0,
        "quiz_score":            48.0,
        "assignment_completion": 55.0,
        "previous_marks":        60.0,
        "_demo_label":           "Needs Attention",
        "_description":          "Low quiz scores and assignment completion flagging early concern.",
    },
    # ── At Risk ───────────────────────────────────────────────────────────
    {
        "student_id":            "DEMO-005",
        "marks":                 32.0,
        "attendance":            45.0,
        "quiz_score":            28.0,
        "assignment_completion": 38.0,
        "previous_marks":        50.0,
        "_demo_label":           "At Risk",
        "_description":          "Very low marks, poor attendance, and declining performance.",
    },
    {
        "student_id":            "DEMO-006",
        "marks":                 41.0,
        "attendance":            52.0,
        "quiz_score":            35.0,
        "assignment_completion": 42.0,
        "previous_marks":        55.0,
        "_demo_label":           "At Risk",
        "_description":          "Multiple academic risk factors present; needs immediate educator attention.",
    },
]


def get_demo_features(student: Dict) -> Dict:
    """
    Extract the five academic feature fields from a demo student record.
    Returns a dict suitable for ML prediction and risk-factor analysis.
    """
    return {
        "marks":                 student["marks"],
        "attendance":            student["attendance"],
        "quiz_score":            student["quiz_score"],
        "assignment_completion": student["assignment_completion"],
        "previous_marks":        student["previous_marks"],
    }


def list_demo_students() -> None:
    """Print a formatted summary of all demonstration students."""
    print("\n" + "=" * 60)
    print("  EduAI – Demonstration Students  (ALL DATA IS SYNTHETIC)")
    print("=" * 60)
    for s in DEMO_STUDENTS:
        print(f"\n  {s['student_id']}  [{s['_demo_label']}]")
        print(f"    {s['_description']}")
        print(f"    Marks: {s['marks']}  |  Attendance: {s['attendance']}%  "
              f"|  Quiz: {s['quiz_score']}  |  "
              f"Assignment: {s['assignment_completion']}%  |  "
              f"Prev Marks: {s['previous_marks']}")
    print()
