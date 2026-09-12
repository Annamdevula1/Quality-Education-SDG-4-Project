# EduAI – AI-Powered Early Warning & Academic Support System
## SDG 4 – Quality Education

EduAI is a complete, professional AI-powered academic early warning system that combines **Machine Learning** and **IBM Granite 4.0 H Small Generative AI** to help educators identify students who may need academic support — before they fall significantly behind.

---

## Project Structure

```
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

```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.  Configure IBM Granite Credentials

Copy `.env.example` to `.env` and fill in your IBM watsonx.ai credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
WATSONX_API_KEY=your-ibm-watsonx-api-key-here
WATSONX_PROJECT_ID=your-ibm-watsonx-project-id-here
WATSONX_REGION=us-south
```

> If credentials are not set, EduAI runs in **Demo Mode** with clearly labelled pre-written fallback responses. The rest of the system works fully without IBM Granite credentials.

### 3. Train the Model

```bash
cd eduai
python train.py
```

This runs all steps 1–10 and saves:
- `outputs/model/eduai_rf_model.joblib`
- `outputs/model/label_encoder.joblib`
- `outputs/eda/*.png` — EDA charts
- `outputs/evaluation/*.png` — Confusion matrix & feature importance

### 4. Launch the Application

```bash
cd eduai
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## System Architecture

```
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
IBM Granite 4.0 H Small Analysis (via IBM watsonx.ai)
           ↓
Explanation + 3 Recommendations + Monitoring Suggestion
           ↓
Professional Streamlit Dashboard
```

---

## AI Component Responsibilities

| Component | Role |
|-----------|------|
| **Random Forest (ML)** | Predicts overall academic risk level |
| **Risk Factor Rules** | Identifies observable academic warning indicators |
| **IBM Granite 4.0 H Small (GenAI)** | Generates explanation, recommendations & monitoring suggestion |

These components are **clearly separated**. IBM Granite receives the ML prediction and risk factors as inputs and generates natural language guidance.

---

## Risk Label Logic (Transparent)

```
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
```

---

## Risk Factor Thresholds (Rule-Based, Transparent)

| Indicator | Threshold |
|-----------|-----------|
| Low marks | < 50 |
| Low attendance | < 65% |
| Low quiz performance | < 50 |
| Low assignment completion | < 60% |
| Declining performance | Current − Previous < −8 |

---

## Ethical Principles

- ✅ **Synthetic data only** — no real student PII
- ✅ **Transparent risk factors** — documented thresholds
- ✅ **AI supports educators** — teachers make final decisions
- ✅ **No sensitive assumptions** — no medical, family, or financial judgments
- ✅ **Secure credentials** — environment variables only, never hard-coded
- ✅ **Risk labels are not permanent** — clearly communicated in UI
- ✅ **Demo mode** — clearly labelled, never misattributed to IBM Granite

---

## IBM Granite Integration



EduAI integrates **IBM Granite 4.0 H Small** through **IBM Cloud watsonx.ai** to generate personalized academic support guidance.

- **Model:** IBM Granite 4.0 H Small
- **Model ID:** `ibm/granite-4-h-small`
- **Platform:** IBM Cloud watsonx.ai
- Generates explanations based on the predicted risk level
- Generates personalized academic recommendations
- Generates monitoring suggestions for educators
- ML prediction and risk factors are provided as inputs to the Granite model
---

## Streamlit Application
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 23 09" src="https://github.com/user-attachments/assets/3062a0b2-ef29-4ae7-9117-a833aa0f9905" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 23 26" src="https://github.com/user-attachments/assets/d9ac56be-6fb7-41ec-8475-ef6b60b4ff32" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 23 41" src="https://github.com/user-attachments/assets/e3b99a1a-f4c4-4ba6-93b9-c2c7c9d46ef6" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 23 50" src="https://github.com/user-attachments/assets/0c79d84c-8290-43c2-95d5-206a6b5c66a4" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 23 59" src="https://github.com/user-attachments/assets/53618fb1-c863-46fe-a55a-56f15ff9b5fe" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 24 12" src="https://github.com/user-attachments/assets/e1ddbf8c-5a77-403e-9330-c3fa60d16c7c" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 24 23" src="https://github.com/user-attachments/assets/02ce14ec-d3cb-4587-9a87-51e6d9576ac0" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 24 49" src="https://github.com/user-attachments/assets/7f513f16-fe79-4c8d-a7ca-0c317788062d" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 35 50" src="https://github.com/user-attachments/assets/64cabdf6-59df-4b7e-8e92-e8bf96516e28" />











## SDG 4 – Quality Education

EduAI supports SDG 4 by helping educators identify students who may need academic support **earlier**, enabling timely, targeted intervention while keeping teachers responsible for all decisions.


## Note:
without the ibm cloud watsonx.ai credentials are not set means the streamlit can be opens in a web browser in a demo mode.
