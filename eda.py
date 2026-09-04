"""
EduAI – Exploratory Data Analysis
===================================
Step 4: Professional, presentation-ready visualizations

All plots are saved to   outputs/eda/   as high-resolution PNGs.
Call run_eda(df) to generate every chart in one step.
"""

import os
import matplotlib
matplotlib.use("Agg")          # headless – safe in all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

from data_processing import FEATURE_COLS, RISK_ORDER

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {
    "On Track":        "#2ecc71",
    "Needs Attention": "#f39c12",
    "At Risk":         "#e74c3c",
}
OUTPUT_DIR = os.path.join("outputs", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 120


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [EDA] saved → {path}")
    return path


# ─── Individual feature distributions ────────────────────────────────────────
def plot_marks_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["marks"], bins=25, color="#3b82d4", edgecolor="white", linewidth=0.6)
    ax.set_title("Distribution of Student Marks", fontweight="bold")
    ax.set_xlabel("Marks (0–100)")
    ax.set_ylabel("Number of Students")
    ax.axvline(df["marks"].mean(), color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Mean: {df['marks'].mean():.1f}")
    ax.legend()
    return _save(fig, "01_marks_distribution")


def plot_attendance_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["attendance"], bins=25, color="#27ae60", edgecolor="white",
            linewidth=0.6)
    ax.set_title("Distribution of Student Attendance", fontweight="bold")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Number of Students")
    ax.axvline(df["attendance"].mean(), color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Mean: {df['attendance'].mean():.1f}%")
    ax.legend()
    return _save(fig, "02_attendance_distribution")


def plot_quiz_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["quiz_score"], bins=25, color="#9b59b6", edgecolor="white",
            linewidth=0.6)
    ax.set_title("Distribution of Quiz Scores", fontweight="bold")
    ax.set_xlabel("Quiz Score (0–100)")
    ax.set_ylabel("Number of Students")
    ax.axvline(df["quiz_score"].mean(), color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Mean: {df['quiz_score'].mean():.1f}")
    ax.legend()
    return _save(fig, "03_quiz_distribution")


def plot_assignment_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["assignment_completion"], bins=25, color="#e67e22",
            edgecolor="white", linewidth=0.6)
    ax.set_title("Distribution of Assignment Completion", fontweight="bold")
    ax.set_xlabel("Assignment Completion (%)")
    ax.set_ylabel("Number of Students")
    ax.axvline(df["assignment_completion"].mean(), color="#e74c3c",
               linestyle="--", linewidth=1.5,
               label=f"Mean: {df['assignment_completion'].mean():.1f}%")
    ax.legend()
    return _save(fig, "04_assignment_distribution")


# ─── Scatter: Previous Marks vs Current Marks ────────────────────────────────
def plot_prev_vs_current_marks(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    for level in RISK_ORDER:
        sub = df[df["risk_level"] == level]
        ax.scatter(sub["previous_marks"], sub["marks"],
                   label=level, color=PALETTE[level],
                   alpha=0.65, s=45, edgecolors="white", linewidth=0.3)
    # perfect-retention diagonal
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, alpha=0.4,
            label="No change line")
    ax.set_title("Previous Marks vs Current Marks by Risk Level",
                 fontweight="bold")
    ax.set_xlabel("Previous Marks")
    ax.set_ylabel("Current Marks")
    ax.legend(title="Risk Level", loc="upper left")
    return _save(fig, "05_prev_vs_current_marks")


# ─── Scatter: Attendance vs Marks ────────────────────────────────────────────
def plot_attendance_vs_marks(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    for level in RISK_ORDER:
        sub = df[df["risk_level"] == level]
        ax.scatter(sub["attendance"], sub["marks"],
                   label=level, color=PALETTE[level],
                   alpha=0.65, s=45, edgecolors="white", linewidth=0.3)
    ax.set_title("Attendance vs Current Marks by Risk Level", fontweight="bold")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Current Marks")
    ax.legend(title="Risk Level", loc="upper left")
    return _save(fig, "06_attendance_vs_marks")


# ─── Correlation Heatmap ─────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # show lower triangle
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "07_correlation_heatmap")


# ─── Risk Level Distribution ─────────────────────────────────────────────────
def plot_risk_distribution(df: pd.DataFrame):
    counts = df["risk_level"].value_counts().reindex(RISK_ORDER)
    colors = [PALETTE[r] for r in RISK_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(RISK_ORDER, counts.values, color=colors, edgecolor="white",
                linewidth=0.6)
    for i, (level, cnt) in enumerate(zip(RISK_ORDER, counts.values)):
        axes[0].text(i, cnt + 2, str(cnt), ha="center", fontweight="bold",
                     fontsize=11)
    axes[0].set_title("Risk Level Distribution (Count)", fontweight="bold")
    axes[0].set_ylabel("Number of Students")
    axes[0].set_xlabel("Risk Level")

    # Pie chart
    axes[1].pie(counts.values, labels=RISK_ORDER, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Risk Level Distribution (Proportion)", fontweight="bold")

    fig.suptitle("Academic Risk Level Distribution – EduAI Dataset",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "08_risk_distribution")


# ─── Master runner ────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> dict:
    """
    Generate all EDA visualizations and return a dict of {chart_name: file_path}.
    """
    print("\n[EDA] Generating visualizations …")
    paths = {}
    paths["marks_dist"]         = plot_marks_distribution(df)
    paths["attendance_dist"]    = plot_attendance_distribution(df)
    paths["quiz_dist"]          = plot_quiz_distribution(df)
    paths["assignment_dist"]    = plot_assignment_distribution(df)
    paths["prev_vs_current"]    = plot_prev_vs_current_marks(df)
    paths["attendance_marks"]   = plot_attendance_vs_marks(df)
    paths["correlation"]        = plot_correlation_heatmap(df)
    paths["risk_distribution"]  = plot_risk_distribution(df)
    print(f"[EDA] All {len(paths)} charts saved to '{OUTPUT_DIR}/'.\n")
    return paths
