"""
EduAI – AI-Powered Early Warning & Academic Support System
============================================================
Steps covered: 16 (professional UI), 17 (dashboard), 18 (UX flow),
               19 (error handling), 20 (complete integration)

SDG 4 – Quality Education

Run with:
    cd eduai
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from collections import Counter

# ── EduAI modules ─────────────────────────────────────────────────────────────
from data_generator      import generate_student_data
from data_processing     import load_and_validate, prepare_features, FEATURE_COLS, RISK_ORDER
from eda                 import run_eda
from ml_model            import train_evaluate_save, load_model, train_model, evaluate_model, MODEL_PATH
from risk_factors        import identify_risk_factors
from granite_integration import generate_granite_response, credentials_available
from analysis_engine     import analyse_student, analyse_dataframe
from demo_students       import DEMO_STUDENTS, get_demo_features

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduAI – Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RISK_COLORS = {
    "On Track":        "#2ecc71",
    "Needs Attention": "#f39c12",
    "At Risk":         "#e74c3c",
}
RISK_ICONS = {
    "On Track":        "🟢",
    "Needs Attention": "🟡",
    "At Risk":         "🔴",
}
RISK_CSS = {
    "On Track":        "on-track",
    "Needs Attention": "needs-attention",
    "At Risk":         "at-risk",
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2c6fad 100%);
    padding: 2rem 2.5rem; border-radius: 12px;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: #ffffff; margin: 0; font-size: 2.2rem; }
.main-header p  { color: #cce4ff; margin: 0.4rem 0 0; font-size: 1rem; }
.main-header .sdg-tag {
    display: inline-block; margin-top: 0.6rem;
    background: rgba(255,255,255,0.15); color: #fff;
    border-radius: 20px; padding: 3px 12px; font-size: 0.82rem;
}

.kpi-card {
    background: #f7f8fa; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1.2rem 1rem; text-align: center;
}
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #1f2328; }
.kpi-label { font-size: 0.82rem; color: #57606a; margin-top: 0.2rem; }

.result-card {
    background: #f7f8fa; border: 1px solid #e5e7eb;
    border-left: 5px solid #3b82d4; border-radius: 8px; padding: 1.2rem;
    margin-bottom: 0.8rem;
}
.result-card.on-track        { border-left-color: #2ecc71; }
.result-card.needs-attention { border-left-color: #f39c12; }
.result-card.at-risk         { border-left-color: #e74c3c; }

.risk-badge-large {
    font-size: 1.25rem; font-weight: 700; padding: 0.6rem 1.2rem;
    border-radius: 8px; text-align: center; display: inline-block;
}
.risk-on-track        { background: #d4edda; color: #155724; }
.risk-needs-attention { background: #fff3cd; color: #856404; }
.risk-at-risk         { background: #f8d7da; color: #721c24; }

.demo-banner {
    background: #fffbea; border: 1px solid #f0d060;
    border-radius: 8px; padding: 0.7rem 1rem;
    font-size: 0.88rem; color: #7a6000; margin-bottom: 1rem;
}
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #1f2328;
    margin: 1.5rem 0 0.8rem;
    border-bottom: 2px solid #e5e7eb; padding-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (defined before any page code)
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value, color: str = "#1f2328"):
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{color};">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_student_result(result: dict):
    """
    Render the complete analysis result for one student.
    Called from Student Analysis, Demo Students, and Student Results pages.
    """
    level     = result.get("predicted_risk_level", "Unknown")
    css_class = RISK_CSS.get(level, "on-track")
    icon      = RISK_ICONS.get(level, "")
    risk_css  = f"risk-{css_class}"

    st.markdown("---")
    st.markdown("### 📊 Analysis Result")

    # ── Row 1: risk badge + metrics ──────────────────────────────────────
    badge_col, info_col = st.columns([1, 3])
    with badge_col:
        st.markdown(
            f'<div class="result-card {css_class}" style="text-align:center;padding:1.5rem;">'
            f'<div style="font-size:2.5rem;">{icon}</div>'
            f'<div class="{risk_css} risk-badge-large" style="margin-top:0.5rem;">{level}</div>'
            f'<div style="color:#57606a;font-size:0.8rem;margin-top:0.4rem;">ML Predicted Risk</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with info_col:
        st.markdown(f"**Student ID:** `{result['student_id']}`")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Marks",         result.get("marks",                 "–"))
        m2.metric("Attendance",    f"{result.get('attendance', '–')}%")
        m3.metric("Quiz Score",    result.get("quiz_score",            "–"))
        m4.metric("Assignment",    f"{result.get('assignment_completion','–')}%")
        m5.metric("Prev Marks",    result.get("previous_marks",        "–"))

    # ── Row 2: Risk Factors ──────────────────────────────────────────────
    st.markdown("#### 📌 Observable Academic Risk Indicators")
    factors = result.get("risk_factors", [])
    if isinstance(factors, str):
        factors = [f.strip() for f in factors.split(";") if f.strip()]

    if factors:
        n_cols = min(len(factors), 3)
        cols_f = st.columns(n_cols)
        for i, f in enumerate(factors):
            cols_f[i % n_cols].warning(f"⚠️  {f}")
    else:
        st.success("✅ No significant academic risk indicators detected.")

    st.caption(
        "These indicators are based on transparent, rule-based thresholds. "
        "They are intended to support educator judgment, not replace it."
    )

    # ── Row 3: IBM Granite AI Section ────────────────────────────────────
    ai_src  = result.get("ai_source", "DEMO")
    is_demo = "DEMO" in str(ai_src).upper()

    if is_demo:
        st.markdown(
            '<div class="demo-banner">'
            '⚠️ <strong>Demo Response:</strong> The AI analysis below was '
            '<strong>NOT generated by IBM Granite</strong>. It is a pre-written '
            'demo response shown because IBM watsonx.ai credentials are not set.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### 🤖 AI Analysis &nbsp; *(Demo – Not IBM Granite)*")
    else:
        st.markdown("#### 🤖 AI Analysis &nbsp; *(IBM Granite – Live)*")

    # Parse errors
    parse_errors = result.get("parse_errors", [])
    if parse_errors:
        for err in parse_errors:
            st.warning(f"⚠️  AI Response Note: {err}")

    tab_exp, tab_rec, tab_mon = st.tabs(
        ["📝 Explanation", "💡 Recommendations", "👁️ Monitoring Suggestion"]
    )
    with tab_exp:
        explanation = result.get("granite_explanation", "")
        if explanation:
            st.markdown(explanation)
        else:
            st.info("No explanation available.")

    with tab_rec:
        recs = [
            result.get("recommendation_1", "(Not available)"),
            result.get("recommendation_2", "(Not available)"),
            result.get("recommendation_3", "(Not available)"),
        ]
        for i, rec in enumerate(recs, 1):
            st.markdown(f"**{i}.** {rec}")

    with tab_mon:
        monitoring = result.get("monitoring_suggestion", "")
        if monitoring:
            st.info(f"📋  {monitoring}")
        else:
            st.info("No monitoring suggestion available.")


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP  (cached – runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising EduAI system …")
def bootstrap():
    """
    1. Generate / load synthetic dataset
    2. Validate and clean data
    3. Train or load ML model
    4. Generate EDA charts
    Returns (df, model, le, metrics, eda_paths)
    """
    os.makedirs("outputs/eda",        exist_ok=True)
    os.makedirs("outputs/model",      exist_ok=True)
    os.makedirs("outputs/evaluation", exist_ok=True)

    # 1. Dataset
    df_raw = generate_student_data()
    df     = load_and_validate(df_raw, verbose=False)
    X, y   = prepare_features(df)

    # 2. ML model
    if os.path.exists(MODEL_PATH):
        model, le = load_model()
        # Evaluate on a fresh stratified split so metrics are available in the UI.
        # We do NOT retrain — the loaded model is used for prediction unchanged.
        _, _, X_train, X_test, y_train, y_test = train_model(X, y)
        metrics = evaluate_model(model, le, X_test, y_test)
        metrics["model"] = model
        metrics["le"]    = le
        metrics["feature_importances"] = dict(
            zip(FEATURE_COLS, model.feature_importances_)
        )
    else:
        metrics = train_evaluate_save(X, y)
        model   = metrics["model"]
        le      = metrics["le"]

    # 3. EDA charts
    eda_paths = run_eda(df)

    return df, model, le, metrics, eda_paths


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 EduAI</h1>
  <p>AI-Powered Early Warning &amp; Academic Support System</p>
  <span class="sdg-tag">🌍 SDG 4 – Quality Education</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading EduAI …"):
    df, model, le, metrics, eda_paths = bootstrap()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduAI")
    st.markdown("*Early Warning & Academic Support*")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Student Analysis",
            "👥 Demo Students",
            "📋 Student Results",
            "📈 EDA Charts",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # IBM Granite status
    if credentials_available():
        st.success("🟢 IBM Granite: Live")
    else:
        st.warning("🟡 IBM Granite: Demo Mode")
        st.caption(
            "Set `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` environment "
            "variables to enable live IBM Granite responses."
        )

    st.markdown("---")
    st.info(
        "📌 **Decision Support Tool**\n\n"
        "EduAI supports educators. "
        "Final decisions always rest with teachers."
    )
    st.caption("All student data is **entirely synthetic**.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<div class="section-title">📊 System Dashboard & Analytics</div>',
                unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────
    total    = len(df)
    n_track  = (df["risk_level"] == "On Track").sum()
    n_attn   = (df["risk_level"] == "Needs Attention").sum()
    n_risk   = (df["risk_level"] == "At Risk").sum()
    accuracy = metrics["accuracy"]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_card("Total Students",  total)
    with k2: kpi_card("🟢 On Track",     n_track,  "#2ecc71")
    with k3: kpi_card("🟡 Needs Attention", n_attn, "#f39c12")
    with k4: kpi_card("🔴 At Risk",      n_risk,   "#e74c3c")
    with k5: kpi_card("Model Accuracy",  f"{accuracy:.1%}")

    st.markdown("---")

    # ── Risk distribution + Feature importance ────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Risk Level Distribution")
        counts = [(df["risk_level"] == r).sum() for r in RISK_ORDER]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(RISK_ORDER, counts,
                      color=[RISK_COLORS[r] for r in RISK_ORDER],
                      edgecolor="white", linewidth=0.7)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, str(cnt),
                    ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel("Number of Students")
        ax.set_xlabel("Risk Level")
        ax.set_title("Academic Risk Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_r:
        st.subheader("Feature Importance (Random Forest)")
        fi = metrics["feature_importances"]
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1]))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(list(fi_sorted.keys()), list(fi_sorted.values()),
                 color="#3b82d4", edgecolor="white")
        ax2.set_xlabel("Importance Score")
        ax2.set_title("ML Feature Importance")
        for i, (k, v) in enumerate(fi_sorted.items()):
            ax2.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")

    # ── Marks, Attendance, Quiz, Assignment distributions ─────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Marks Distribution by Risk Level")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        for level in RISK_ORDER:
            ax3.hist(df[df["risk_level"] == level]["marks"],
                     bins=20, alpha=0.65, label=level,
                     color=RISK_COLORS[level], edgecolor="white")
        ax3.set_xlabel("Marks")
        ax3.set_ylabel("Students")
        ax3.set_title("Marks Distribution")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_b:
        st.subheader("Attendance Distribution by Risk Level")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        for level in RISK_ORDER:
            ax4.hist(df[df["risk_level"] == level]["attendance"],
                     bins=20, alpha=0.65, label=level,
                     color=RISK_COLORS[level], edgecolor="white")
        ax4.set_xlabel("Attendance (%)")
        ax4.set_ylabel("Students")
        ax4.set_title("Attendance Distribution")
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Quiz Score Distribution")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        for level in RISK_ORDER:
            ax5.hist(df[df["risk_level"] == level]["quiz_score"],
                     bins=20, alpha=0.65, label=level,
                     color=RISK_COLORS[level], edgecolor="white")
        ax5.set_xlabel("Quiz Score")
        ax5.set_ylabel("Students")
        ax5.set_title("Quiz Performance Distribution")
        ax5.legend()
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    with col_d:
        st.subheader("Assignment Completion Distribution")
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        for level in RISK_ORDER:
            ax6.hist(df[df["risk_level"] == level]["assignment_completion"],
                     bins=20, alpha=0.65, label=level,
                     color=RISK_COLORS[level], edgecolor="white")
        ax6.set_xlabel("Assignment Completion (%)")
        ax6.set_ylabel("Students")
        ax6.set_title("Assignment Completion Distribution")
        ax6.legend()
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    st.markdown("---")

    # ── Current vs Previous marks + Risk Factor Frequency ─────────────────
    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Current Marks vs Previous Marks")
        fig7, ax7 = plt.subplots(figsize=(6, 5))
        for level in RISK_ORDER:
            sub = df[df["risk_level"] == level]
            ax7.scatter(sub["previous_marks"], sub["marks"],
                        label=level, color=RISK_COLORS[level],
                        alpha=0.55, s=28, edgecolors="white", linewidth=0.3)
        ax7.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1,
                 label="No change")
        ax7.set_xlabel("Previous Marks")
        ax7.set_ylabel("Current Marks")
        ax7.set_title("Marks Trajectory by Risk Level")
        ax7.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

    with col_f:
        st.subheader("Risk Factor Frequency")
        all_factors = []
        for _, row in df.iterrows():
            all_factors.extend(identify_risk_factors(
                row["marks"], row["attendance"], row["quiz_score"],
                row["assignment_completion"], row["previous_marks"]
            ))
        fc = Counter(all_factors)
        if fc:
            fig8, ax8 = plt.subplots(figsize=(6, 5))
            labels_f = list(fc.keys())
            vals_f   = list(fc.values())
            sorted_idx = sorted(range(len(vals_f)), key=lambda i: vals_f[i])
            ax8.barh([labels_f[i] for i in sorted_idx],
                     [vals_f[i]   for i in sorted_idx],
                     color="#7c5cd8", edgecolor="white")
            ax8.set_xlabel("Number of Students")
            ax8.set_title("Risk Factor Frequency (All Students)")
            plt.tight_layout()
            st.pyplot(fig8)
            plt.close(fig8)

    # ── Students Requiring Attention ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚨 Students Requiring Attention")
    st.caption("Students classified as **Needs Attention** or **At Risk**.")

    attention_df = df[df["risk_level"].isin(["At Risk", "Needs Attention"])][
        ["student_id", "marks", "attendance", "quiz_score",
         "assignment_completion", "previous_marks", "risk_level"]
    ].sort_values("risk_level").reset_index(drop=True)

    def _color_risk(val):
        if val == "At Risk":
            return "background-color: #f8d7da; color: #721c24;"
        if val == "Needs Attention":
            return "background-color: #fff3cd; color: #856404;"
        return ""

    try:
        styled = attention_df.style.map(_color_risk, subset=["risk_level"])
    except AttributeError:
        styled = attention_df.style.applymap(_color_risk, subset=["risk_level"])

    st.dataframe(styled, use_container_width=True, height=380)
    st.caption(f"Showing {len(attention_df)} of {total} students. All data is synthetic.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Student Analysis":
    st.markdown('<div class="section-title">🔍 Individual Student Analysis</div>',
                unsafe_allow_html=True)
    st.caption(
        "Enter a student's academic indicators to run the complete EduAI pipeline. "
        "All data is for demonstration purposes only."
    )

    if not credentials_available():
        st.markdown(
            '<div class="demo-banner">⚠️ <strong>Demo Mode:</strong> '
            'IBM Granite credentials not set. AI responses below are '
            '<strong>pre-written demos — NOT generated by IBM Granite</strong>.'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.form("student_form"):
        st.markdown("#### Enter Academic Indicators")
        c1, c2, c3 = st.columns(3)

        with c1:
            s_id   = st.text_input("Student ID", value="STU-CUSTOM-001")
            marks  = st.slider("Current Marks",  0.0, 100.0, 65.0, 0.5)

        with c2:
            attendance = st.slider("Attendance (%)",             0.0, 100.0, 75.0, 0.5)
            quiz_score = st.slider("Quiz Score",                 0.0, 100.0, 60.0, 0.5)

        with c3:
            assignment = st.slider("Assignment Completion (%)",  0.0, 100.0, 70.0, 0.5)
            prev_marks = st.slider("Previous Marks",             0.0, 100.0, 68.0, 0.5)

        submitted = st.form_submit_button(
            "🔍 Run Analysis", type="primary", use_container_width=True
        )

    if submitted:
        with st.spinner("Running EduAI analysis …"):
            try:
                result = analyse_student(
                    student_id=s_id.strip() or "CUSTOM-001",
                    marks=marks,
                    attendance=attendance,
                    quiz_score=quiz_score,
                    assignment_completion=assignment,
                    previous_marks=prev_marks,
                    model=model,
                    le=le,
                )
                render_student_result(result)
            except Exception as e:
                st.error(f"⛔ Analysis error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMO STUDENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demo Students":
    st.markdown('<div class="section-title">👥 Demonstration Students</div>',
                unsafe_allow_html=True)
    st.info(
        "⚠️  **All demonstration students are entirely synthetic.** "
        "No real student information is used anywhere in this system."
    )

    if not credentials_available():
        st.markdown(
            '<div class="demo-banner">⚠️ <strong>Demo Mode:</strong> '
            'AI responses are pre-written demos — '
            '<strong>NOT generated by IBM Granite</strong>.</div>',
            unsafe_allow_html=True,
        )

    for demo in DEMO_STUDENTS:
        icon  = RISK_ICONS.get(demo["_demo_label"], "")
        label = demo["_demo_label"]
        title = f"{icon}  {demo['student_id']}  ·  {label}  —  {demo['_description']}"

        with st.expander(title, expanded=False):
            st.caption("All data shown is synthetic.")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Marks",          demo["marks"])
            m2.metric("Attendance",     f"{demo['attendance']}%")
            m3.metric("Quiz Score",     demo["quiz_score"])
            m4.metric("Assignment",     f"{demo['assignment_completion']}%")
            m5.metric("Prev Marks",     demo["previous_marks"])

            result_key = f"demo_result_{demo['student_id']}"

            if st.button(f"▶  Analyse {demo['student_id']}",
                         key=f"btn_{demo['student_id']}", type="primary"):
                with st.spinner(f"Analysing {demo['student_id']} …"):
                    try:
                        res = analyse_student(
                            student_id=demo["student_id"],
                            **get_demo_features(demo),
                            model=model,
                            le=le,
                        )
                        st.session_state[result_key] = res
                    except Exception as e:
                        st.session_state[result_key] = {"error": str(e)}

            saved = st.session_state.get(result_key)
            if saved:
                if "error" in saved:
                    st.error(f"Analysis error: {saved['error']}")
                else:
                    render_student_result(saved)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDENT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Student Results":
    st.markdown('<div class="section-title">📋 Full Student Results Table</div>',
                unsafe_allow_html=True)
    st.caption(
        "Predicted risk levels for all 500 synthetic students. "
        "Click 'Run Full Analysis' to include AI explanations (takes ~1–2 min)."
    )

    # Summary table (fast – no Granite)
    summary_cols = [
        "student_id", "marks", "attendance", "quiz_score",
        "assignment_completion", "previous_marks", "risk_level",
    ]
    summary_df = df[summary_cols].copy()
    summary_df.columns = [
        "Student ID", "Marks", "Attendance (%)", "Quiz Score",
        "Assignment (%)", "Previous Marks", "Risk Level",
    ]

    # Filter controls
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_level = st.selectbox(
            "Filter by Risk Level",
            ["All Students", "On Track", "Needs Attention", "At Risk"]
        )
    with col_f2:
        search_id = st.text_input("Search Student ID", "")

    filtered = summary_df.copy()
    if filter_level != "All Students":
        filtered = filtered[filtered["Risk Level"] == filter_level]
    if search_id.strip():
        filtered = filtered[
            filtered["Student ID"].str.contains(search_id.strip(), case=False)
        ]

    st.caption(f"Showing **{len(filtered)}** of **{len(summary_df)}** students.")

    def _color(val):
        if val == "At Risk":         return "background-color:#f8d7da;color:#721c24;"
        if val == "Needs Attention": return "background-color:#fff3cd;color:#856404;"
        if val == "On Track":        return "background-color:#d4edda;color:#155724;"
        return ""

    try:
        styled = filtered.style.map(_color, subset=["Risk Level"])
    except AttributeError:
        styled = filtered.style.applymap(_color, subset=["Risk Level"])

    st.dataframe(styled, use_container_width=True, height=420)

    # ── Full Analysis (with Granite) ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 Full AI Analysis")

    run_full = st.button("▶  Run Full Analysis on Selected Students",
                         type="primary")

    if run_full or "full_results" in st.session_state:
        if run_full and "full_results" not in st.session_state:
            students_to_analyse = df[
                df["student_id"].isin(filtered["Student ID"].tolist())
            ].head(50)  # limit to 50 for demo performance

            if len(students_to_analyse) > 0:
                progress_bar = st.progress(0)
                status_text  = st.empty()

                results_list = []
                total_s = len(students_to_analyse)

                for idx_s, (_, row) in enumerate(students_to_analyse.iterrows()):
                    status_text.text(f"Analysing {row['student_id']} ({idx_s+1}/{total_s}) …")
                    try:
                        res = analyse_student(
                            student_id=row["student_id"],
                            marks=float(row["marks"]),
                            attendance=float(row["attendance"]),
                            quiz_score=float(row["quiz_score"]),
                            assignment_completion=float(row["assignment_completion"]),
                            previous_marks=float(row["previous_marks"]),
                            model=model,
                            le=le,
                        )
                        results_list.append(res)
                    except Exception as e:
                        results_list.append({
                            "student_id": row["student_id"],
                            "error": str(e),
                        })
                    progress_bar.progress((idx_s + 1) / total_s)

                st.session_state["full_results"] = pd.DataFrame(results_list)
                progress_bar.empty()
                status_text.empty()

        full_res = st.session_state.get("full_results")
        if full_res is not None and len(full_res) > 0:
            st.success(f"Analysis complete for {len(full_res)} students.")

            # Detailed view
            sel_id = st.selectbox(
                "Select a student to view full analysis",
                full_res["student_id"].tolist()
            )
            if sel_id:
                sel_row = full_res[full_res["student_id"] == sel_id].iloc[0].to_dict()
                render_student_result(sel_row)

            # Download
            st.markdown("---")
            dl_df = full_res.copy()
            if "risk_factors" in dl_df.columns:
                dl_df["risk_factors"] = dl_df["risk_factors"].apply(
                    lambda x: "; ".join(x) if isinstance(x, list) else str(x)
                )
            csv = dl_df.to_csv(index=False)
            st.download_button(
                "⬇️  Download Full Results CSV",
                data=csv,
                file_name="eduai_full_results.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Charts":
    st.markdown('<div class="section-title">📈 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)
    st.caption("Professional visualizations of the EduAI synthetic student dataset.")

    chart_configs = [
        ("marks_dist",        "Marks Distribution"),
        ("attendance_dist",   "Attendance Distribution"),
        ("quiz_dist",         "Quiz Score Distribution"),
        ("assignment_dist",   "Assignment Completion Distribution"),
        ("prev_vs_current",   "Previous vs Current Marks"),
        ("attendance_marks",  "Attendance vs Marks"),
        ("correlation",       "Feature Correlation Heatmap"),
        ("risk_distribution", "Risk Level Distribution"),
    ]

    file_map = {
        "marks_dist":        "01_marks_distribution",
        "attendance_dist":   "02_attendance_distribution",
        "quiz_dist":         "03_quiz_distribution",
        "assignment_dist":   "04_assignment_distribution",
        "prev_vs_current":   "05_prev_vs_current_marks",
        "attendance_marks":  "06_attendance_vs_marks",
        "correlation":       "07_correlation_heatmap",
        "risk_distribution": "08_risk_distribution",
    }

    for i in range(0, len(chart_configs), 2):
        col_l, col_r = st.columns(2)
        pair = chart_configs[i:i+2]
        for col, (key, title) in zip([col_l, col_r], pair):
            img_path = f"outputs/eda/{file_map[key]}.png"
            with col:
                st.subheader(title)
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.info("Chart not yet generated.")

    # Confusion matrix and feature importance
    st.markdown("---")
    st.markdown("### Model Evaluation Charts")
    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix")
        cm_path = "outputs/evaluation/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Run the model training to generate the confusion matrix.")
    with col_fi:
        st.subheader("Feature Importance")
        fi_path = "outputs/evaluation/feature_importance.png"
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Run the model training to generate the feature importance chart.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="section-title">ℹ️ About EduAI</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### What is EduAI?
EduAI is a **complete AI-powered academic early warning and support system**
aligned with **SDG 4: Quality Education**.

It combines **Machine Learning** and **IBM Granite Generative AI** to help
educators identify students who may need academic attention — earlier.

---
### System Pipeline

**1. Synthetic Data (500 students)**
Reproducible, anonymized dataset. No real student PII.

**2. Data Validation & EDA**
Full pipeline: validation, cleaning, descriptive statistics,
8 professional visualizations.

**3. Machine Learning (Random Forest)**
Classifies students into three risk levels:
- 🟢 On Track
- 🟡 Needs Attention
- 🔴 At Risk

**4. Transparent Risk Factors (Rule-Based)**
Separate from ML. Observable academic indicators with clear thresholds.

**5. IBM Granite AI Guidance**
Generates personalized explanation, 3 recommendations,
and 1 monitoring suggestion per student.
        """)

    with col2:
        st.markdown("""
### AI Component Responsibilities

| Component | Role |
|-----------|------|
| **Random Forest** | Predicts overall academic risk level |
| **Risk Factor Rules** | Identifies observable academic indicators |
| **IBM Granite** | Generates explanation, recommendations & monitoring |

---
### Ethical Principles
- ✅ Synthetic data only — no real student PII
- ✅ Transparent, rule-based risk factors with documented thresholds
- ✅ AI supports educators — teachers make final decisions
- ✅ No sensitive personal, medical, or family assumptions
- ✅ Secure credential management (env vars only)
- ✅ Risk labels are not permanent labels
- ✅ No medical or psychological diagnoses

---
### Technical Stack
| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| ML | Scikit-learn RandomForest |
| Generative AI | IBM Granite via IBM watsonx.ai |
| Visualization | Matplotlib, Seaborn |
| Data | Pandas, NumPy |
| Model Persistence | Joblib |
| Language | Python 3.10+ |
        """)

    st.markdown("---")
    st.markdown("""
### Risk Factor Thresholds (Transparent)

| Indicator | Threshold |
|-----------|-----------|
| Low marks | < 50 marks |
| Low attendance | < 65% |
| Low quiz performance | < 50 score |
| Low assignment completion | < 60% |
| Declining performance | Current marks < Previous marks − 8 |
    """)
    st.caption(
        "EduAI Prototype · SDG 4 – Quality Education · "
        "All student data is entirely synthetic · "
        "IBM Granite powered by IBM watsonx.ai"
    )
