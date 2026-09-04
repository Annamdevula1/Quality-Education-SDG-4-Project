"""
EduAI – Machine Learning Risk Classification
=============================================
Steps covered: 5 (training), 6 (evaluation), 10 (model saving)

Model: Random Forest Classifier
Serialization: joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from data_processing import FEATURE_COLS, RISK_ORDER

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join("outputs", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODEL_DIR, "eduai_rf_model.joblib")
ENCODER_PATH  = os.path.join(MODEL_DIR, "label_encoder.joblib")
OUTPUT_DIR    = os.path.join("outputs", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20     # 80/20 split


# ─── Training ─────────────────────────────────────────────────────────────────
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X : pd.DataFrame   – feature matrix (FEATURE_COLS)
    y : pd.Series      – risk_level labels

    Returns
    -------
    model        : fitted RandomForestClassifier
    le           : fitted LabelEncoder
    X_train, X_test, y_train, y_test  (for immediate evaluation)
    """
    # Encode labels
    le = LabelEncoder()
    le.fit(RISK_ORDER)        # fixed order: On Track, Needs Attention, At Risk
    y_enc = le.transform(y)

    # Stratified train/test split (ensures all three classes in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    # Model – no feature preprocessing needed for Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    print(f"[ML] Model trained on {len(X_train)} samples.")
    print(f"[ML] Test set size: {len(X_test)} samples.")

    return model, le, X_train, X_test, y_train, y_test


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, le, X_test, y_test) -> dict:
    """
    Evaluate the trained model on the test set.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, report, conf_matrix
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    target_names = le.classes_.tolist()

    report = classification_report(
        y_test, y_pred, target_names=target_names, digits=3
    )

    conf_mat = confusion_matrix(y_test, y_pred)

    macro_p  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r  = recall_score(y_test, y_pred,    average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred,        average="macro", zero_division=0)

    print("\n[ML] ── Classification Report ──────────────────────────────")
    print(report)
    print(f"[ML] Accuracy       : {acc:.4f}")
    print(f"[ML] Macro Precision: {macro_p:.4f}")
    print(f"[ML] Macro Recall   : {macro_r:.4f}")
    print(f"[ML] Macro F1-Score : {macro_f1:.4f}")

    return {
        "accuracy":   acc,
        "precision":  macro_p,
        "recall":     macro_r,
        "f1":         macro_f1,
        "report":     report,
        "conf_matrix": conf_mat,
        "classes":    target_names,
    }


# ─── Confusion Matrix Visualization ──────────────────────────────────────────
def plot_confusion_matrix(conf_mat: np.ndarray, class_names: list) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        conf_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax, annot_kws={"size": 13},
    )
    ax.set_title("Confusion Matrix – EduAI Risk Classifier", fontweight="bold",
                 fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Confusion matrix saved → {path}")
    return path


# ─── Feature Importance Visualization ────────────────────────────────────────
def plot_feature_importance(model, feature_names: list) -> str:
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names[::-1], sorted_vals[::-1],
                   color="#3b82d4", edgecolor="white", linewidth=0.5)
    ax.set_title("Random Forest – Feature Importance", fontweight="bold",
                 fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Academic Feature")
    for bar, val in zip(bars, sorted_vals[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Feature importance chart saved → {path}")
    return path


# ─── Save / Load ──────────────────────────────────────────────────────────────
def save_model(model, le):
    """Persist the trained model and label encoder to disk."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le,    ENCODER_PATH)
    print(f"[ML] Model saved    → {MODEL_PATH}")
    print(f"[ML] Encoder saved  → {ENCODER_PATH}")


def load_model():
    """
    Load the persisted model and label encoder.

    Returns
    -------
    model : RandomForestClassifier
    le    : LabelEncoder
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Run train_evaluate_save() first (or run train.py)."
        )
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(ENCODER_PATH)
    print(f"[ML] Model loaded from '{MODEL_PATH}'.")
    return model, le


# ─── Predict single student ───────────────────────────────────────────────────
def predict_single(model, le, student_features: dict) -> str:
    """
    Predict risk level for a single student.

    Parameters
    ----------
    model            : fitted RandomForestClassifier
    le               : fitted LabelEncoder
    student_features : dict with keys matching FEATURE_COLS

    Returns
    -------
    str  – one of "On Track", "Needs Attention", "At Risk"
    """
    X = pd.DataFrame([student_features])[FEATURE_COLS]
    y_enc = model.predict(X)[0]
    return le.inverse_transform([y_enc])[0]


# ─── Convenience: train + evaluate + save in one call ─────────────────────────
def train_evaluate_save(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Full pipeline: train → evaluate → save model.

    Returns the evaluation metrics dict (same as evaluate_model).
    """
    model, le, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics = evaluate_model(model, le, X_test, y_test)

    # Confusion matrix + feature importance charts
    metrics["conf_matrix_path"]      = plot_confusion_matrix(
        metrics["conf_matrix"], metrics["classes"]
    )
    metrics["feature_importance_path"] = plot_feature_importance(
        model, FEATURE_COLS
    )
    metrics["feature_importances"] = dict(
        zip(FEATURE_COLS, model.feature_importances_)
    )

    save_model(model, le)
    metrics["model"] = model
    metrics["le"]    = le
    return metrics
