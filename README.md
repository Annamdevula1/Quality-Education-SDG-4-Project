EduAI – AI-Powered Early Warning & Academic Support System

SDG 4 – Quality Education

EduAI is a complete, professional AI-powered academic early warning system that combines Machine Learning and IBM Granite Generative AI to help educators identify students who may need academic support — before they fall significantly behind.

⸻

Project Structure

eduai/

├── app.py                  ← Streamlit web application (Steps 16–20)

├── train.py                ← Training & evaluation script (Steps 1–10)

├── data_generator.py       ← Synthetic dataset generation (Steps 1, 3)

├── data_processing.py      ← Validation & processing pipeline (Step 2)

├── eda.py                  ← Exploratory data analysis (Step 4)

├── ml_model.py             ← Random Forest ML model (Steps 5, 6, 10)

├── risk_factors.py         ← Transparent rule-based indicators (Step 7)

├── analysis_engine.py      ← Single & batch student analysis (Steps 8, 15)

├── granite_integration.py  ← IBM Granite via watsonx.ai (Steps 11–14)

├── demo_students.py        ← Predefined synthetic demo students (Step 9)

├── requirements.txt        ← Python dependencies

├── .env.example            ← Credentials template


⸻

Quick Start

1. Install Dependencies

pip install -r requirements.txt

2. Configure IBM Granite Credentials

Copy .env.example to .env and fill in your IBM watsonx.ai credentials:

WATSONX_API_KEY=your-ibm-watsonx-api-key-here

WATSONX_PROJECT_ID=your-ibm-watsonx-project-id-here

WATSONX_REGION=us-south

3. Train the Model

python train.py

This runs all steps 1–10 and saves:

* outputs/model/eduai_rf_model.joblib
* outputs/model/label_encoder.joblib
* outputs/eda/*.png — EDA charts
* outputs/evaluation/*.png — Confusion matrix & feature importance

4. Launch the Application

streamlit run app.py

open local host in a web browser.

⸻

System Architecture

Synthetic Student Data (500 records)

           ↓
Data Validation & Processing

           ↓
Risk Label Creation (transparent rules)

           ↓
Exploratory Data Analysis (8 charts)

           ↓
Random Forest ML Training (80/20 split, stratified)

           ↓
Risk Classification: On Track / Needs Attention / At Risk

           ↓
Transparent Risk Factor Identification (rule-based)


           ↓
IBM Granite Analysis (via IBM watsonx.ai)


           ↓
Explanation + 3 Recommendations + Monitoring Suggestion

           ↓
Professional Streamlit Dashboard


⸻

AI Component Responsibilities

Component	Role
Random Forest (ML)	Predicts overall academic risk level
Risk Factor Rules	Identifies observable academic warning indicators
IBM Granite (GenAI)	Generates explanation, recommendations & monitoring suggestion

These components are clearly separated. IBM Granite receives the ML prediction and risk factors as inputs and generates natural language guidance.

⸻

Risk Label Logic (Transparent)

At Risk:
  - marks < 45, OR
  - attendance < 55%, OR
  - (quiz_score < 40 AND assignment_completion < 50%)
Needs Attention:
  - marks < 60, OR
  - attendance < 70%, OR
  - quiz_score < 55, OR
  - assignment_completion < 60%, OR
  - current marks − previous marks < −10
On Track:
  - All other students

⸻

Risk Factor Thresholds (Rule-Based, Transparent)

Indicator	Threshold
Low marks	< 50
Low attendance	< 65%
Low quiz performance	< 50
Low assignment completion	< 60%
Declining performance	Current − Previous < −8

⸻

Ethical Principles

* ✅ Synthetic data only — no real student PII
* ✅ Transparent risk factors — documented thresholds
* ✅ AI supports educators — teachers make final decisions
* ✅ No sensitive assumptions — no medical, family, or financial judgments
* ✅ Secure credentials — environment variables only, never hard-coded
* ✅ Risk labels are not permanent — clearly communicated in UI

⸻

IBM Granite Integration

EduAI is successfully integrated with IBM Granite through IBM watsonx.ai.

The application sends the Machine Learning risk prediction and identified academic risk factors to IBM Granite.

IBM Granite generates:

* A clear explanation of the student’s academic situation
* Three academic support recommendations
* A monitoring or follow-up suggestion

IBM Granite does not independently assign the student’s risk category. The Random Forest model performs the risk classification, while IBM Granite generates natural-language guidance based on the prediction and observable academic indicators.

Note:

-the project can be works in a demo mode also without credentails also

⸻

SDG 4 – Quality Education

EduAI supports SDG 4 by helping educators identify students who may need academic support earlier, enabling timely, targeted intervention while keeping teachers responsible for all decisions.
