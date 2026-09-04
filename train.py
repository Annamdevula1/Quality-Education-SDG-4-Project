"""
EduAI – Training Script
=========================
Run this script once to train and save the ML model before launching the app.

Usage:
    cd eduai
    python train.py

This script covers Steps 1–10 of the EduAI project specification:
  1.  Generate 500-record synthetic student dataset
  2.  Validate and process data
  3.  Risk label creation (already embedded in data generator)
  4.  EDA visualizations
  5.  Random Forest ML training
  6.  Model evaluation (accuracy, confusion matrix, feature importance)
  7.  Risk factor identification (demonstrated on dataset)
  8.  Individual student analysis examples
  9.  Demonstration student analysis
  10. Model saved to outputs/model/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from data_generator   import generate_student_data
from data_processing  import load_and_validate, prepare_features, RISK_ORDER
from eda              import run_eda
from ml_model         import train_evaluate_save, predict_single
from risk_factors     import identify_risk_factors, build_risk_factor_summary
from analysis_engine  import analyse_student
from demo_students    import DEMO_STUDENTS, get_demo_features, list_demo_students


def main():
    print("\n" + "=" * 65)
    print("  EduAI – Training & Evaluation Script")
    print("  SDG 4 – Quality Education")
    print("=" * 65)

    # ── Step 1: Generate synthetic dataset ────────────────────────────────
    print("\n[Step 1] Generating synthetic student dataset (500 records) …")
    df_raw = generate_student_data(n_students=500, seed=42)
    print(f"  Generated {len(df_raw)} student records.")
    print(f"  Columns: {list(df_raw.columns)}")

    # ── Step 2: Data validation & processing ──────────────────────────────
    print("\n[Step 2] Running data validation & processing pipeline …")
    df = load_and_validate(df_raw, verbose=True)

    # ── Step 3: Risk label distribution ───────────────────────────────────
    print("\n[Step 3] Risk label distribution:")
    for level in RISK_ORDER:
        n   = (df["risk_level"] == level).sum()
        pct = 100 * n / len(df)
        print(f"  {level:<18} {n:>4}  ({pct:.1f}%)")

    # ── Step 4: EDA visualizations ────────────────────────────────────────
    print("\n[Step 4] Generating EDA visualizations …")
    os.makedirs("outputs/eda", exist_ok=True)
    eda_paths = run_eda(df)

    # ── Steps 5 & 6: ML training + evaluation ─────────────────────────────
    print("\n[Steps 5 & 6] Training Random Forest + evaluating model …")
    os.makedirs("outputs/model",      exist_ok=True)
    os.makedirs("outputs/evaluation", exist_ok=True)

    X, y    = prepare_features(df)
    metrics = train_evaluate_save(X, y)

    print(f"\n  Model Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro Precision  : {metrics['precision']:.4f}")
    print(f"  Macro Recall     : {metrics['recall']:.4f}")
    print(f"  Macro F1-Score   : {metrics['f1']:.4f}")

    print("\n  Feature Importances:")
    for feat, imp in sorted(metrics["feature_importances"].items(),
                             key=lambda x: x[1], reverse=True):
        print(f"    {feat:<26} {imp:.4f}")

    # ── Step 7: Rule-based risk factor demonstration ───────────────────────
    print("\n[Step 7] Risk Factor Identification – sample students:")
    sample = df.sample(5, random_state=42)
    for _, row in sample.iterrows():
        factors = identify_risk_factors(
            row["marks"], row["attendance"], row["quiz_score"],
            row["assignment_completion"], row["previous_marks"]
        )
        level = predict_single(metrics["model"], metrics["le"], {
            "marks":                 row["marks"],
            "attendance":            row["attendance"],
            "quiz_score":            row["quiz_score"],
            "assignment_completion": row["assignment_completion"],
            "previous_marks":        row["previous_marks"],
        })
        summary = build_risk_factor_summary(row["student_id"], level, factors)
        print()
        print(summary)

    # ── Step 8: Individual student analysis ───────────────────────────────
    print("\n[Step 8] Individual student analysis (custom example):")
    custom = analyse_student(
        student_id="TRAIN-DEMO",
        marks=38.0,
        attendance=48.0,
        quiz_score=32.0,
        assignment_completion=40.0,
        previous_marks=55.0,
        model=metrics["model"],
        le=metrics["le"],
    )
    print(f"  Student ID       : {custom['student_id']}")
    print(f"  Predicted Risk   : {custom['predicted_risk_level']}")
    print(f"  Risk Factors     : {custom['risk_factors']}")
    print(f"  AI Source        : {custom['ai_source']}")
    print(f"  Explanation      : {custom['granite_explanation'][:120]} …")
    print(f"  Recommendation 1 : {custom['recommendation_1'][:80]} …")

    # ── Step 9: Demonstration students ────────────────────────────────────
    print("\n[Step 9] Demonstration Students:")
    list_demo_students()
    print("  Running analysis on all demonstration students …")
    for demo in DEMO_STUDENTS:
        result = analyse_student(
            student_id=demo["student_id"],
            **get_demo_features(demo),
            model=metrics["model"],
            le=metrics["le"],
        )
        print(f"  {demo['student_id']}  →  {result['predicted_risk_level']}"
              f"  (expected: {demo['_demo_label']})")

    # ── Step 10: Model saved confirmation ─────────────────────────────────
    print("\n[Step 10] Model persistence:")
    print("  outputs/model/eduai_rf_model.joblib  – Random Forest model")
    print("  outputs/model/label_encoder.joblib   – Label encoder")
    print("  EDA charts                           – outputs/eda/")
    print("  Evaluation charts                    – outputs/evaluation/")

    print("\n" + "=" * 65)
    print("  EduAI training complete!")
    print("  Launch the app with:  streamlit run app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
