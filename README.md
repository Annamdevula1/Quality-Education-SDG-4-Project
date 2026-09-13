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
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 36 33" src="https://github.com/user-attachments/assets/56932b91-43a0-4293-8ac1-6e6980bdd3d5" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 36 44" src="https://github.com/user-attachments/assets/9a3b0337-f65b-46c5-992d-5d70b65620bc" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 36 51" src="https://github.com/user-attachments/assets/c13a65e8-0119-4f04-b25a-8a79c94f04ee" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 38 25" src="https://github.com/user-attachments/assets/9b1cd6b6-9d19-4474-aaea-a2b8b1242416" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 38 25" src="https://github.com/user-attachments/assets/5891ab51-fc8c-432b-9f78-e4b0d3ad2b68" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 38 57" src="https://github.com/user-attachments/assets/0144cb54-10b3-476a-bb9d-624323664751" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 39 12" src="https://github.com/user-attachments/assets/3eb48489-e180-485c-b757-a2f60cdb7066" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 39 21" src="https://github.com/user-attachments/assets/a46cc756-4211-480d-be63-2bed336d7370" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 41 03" src="https://github.com/user-attachments/assets/275d5c57-d953-4e77-b54e-3ef000109fe8" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 41 32" src="https://github.com/user-attachments/assets/ad42ccd2-fb2b-46f6-8c21-ba3d0b9149d2" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 41 44" src="https://github.com/user-attachments/assets/013d64dd-675e-4ccd-b778-7ff26e81f8f9" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 41 53" src="https://github.com/user-attachments/assets/4d65a11b-041b-474c-be9f-cf0c708339ed" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 42 06" src="https://github.com/user-attachments/assets/7741b9f0-df8a-47ba-bb41-15de51cc5174" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 42 35" src="https://github.com/user-attachments/assets/6bcd0069-14ff-4b88-96c6-edab26fdb67e" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 36 44" src="https://github.com/user-attachments/assets/4843d469-3c58-4851-8e06-009d5c98d5dc" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 43 06" src="https://github.com/user-attachments/assets/3882fb4e-fad4-4189-ae0f-3416bba46024" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 43 15" src="https://github.com/user-attachments/assets/688ec313-4767-4c9f-8b49-e4d31547316b" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 43 25" src="https://github.com/user-attachments/assets/f2e858d1-b64d-4def-b27b-3983a429c6c5" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 43 25" src="https://github.com/user-attachments/assets/90c3a541-37b5-46a3-9176-8ac3ebd0b3b9" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 43 41" src="https://github.com/user-attachments/assets/fdb9119d-cd09-4585-a21c-6337221c1677" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 44 21" src="https://github.com/user-attachments/assets/b7fa4954-a292-4706-9216-1db1e93b28f4" />
<img width="1920" height="1080" alt="Screenshot 2026-09-10 17 44 30" src="https://github.com/user-attachments/assets/7ad050ae-67b3-4d8d-ae95-0cade19e98e4" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 19 16" src="https://github.com/user-attachments/assets/97e1c7bb-09cc-4622-9bc9-eced935e311f" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 19 27" src="https://github.com/user-attachments/assets/eef5b2ef-c8db-435d-a8a5-90777e5ebdc3" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 19 27" src="https://github.com/user-attachments/assets/c251aa53-3b9a-4b23-bb8b-755c10e70f41" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 19 37" src="https://github.com/user-attachments/assets/1db54d6c-f6b6-4358-b4e1-007ed00c51ee" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 20 00" src="https://github.com/user-attachments/assets/7421c5ad-d001-4998-b696-39058c78ccf6" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 20 15" src="https://github.com/user-attachments/assets/92e77a51-fd21-447b-9488-0a8cc33868f9" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 20 15" src="https://github.com/user-attachments/assets/ee4f9ceb-1e92-4fd9-a7a8-470b6da9123b" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 21 01" src="https://github.com/user-attachments/assets/ad28adf9-d616-4c09-b0d8-8d80e584a704" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 21 01" src="https://github.com/user-attachments/assets/3e154a70-edcb-48d1-b25b-6929487f206d" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 21 30" src="https://github.com/user-attachments/assets/69cd0f19-ce7f-488e-943b-b3913bc5b6ea" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 21 30" src="https://github.com/user-attachments/assets/5c5c246d-6ebe-483f-af88-487d3a376f7d" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 21 43" src="https://github.com/user-attachments/assets/e4672615-e4fd-4e4a-ba4e-effd3583142e" />
<img width="1920" height="1080" alt="Screenshot 2026-09-12 15 19 37" src="https://github.com/user-attachments/assets/4bccf3fe-7149-4013-b188-8e92b3374cde" />








































## SDG 4 – Quality Education

EduAI supports SDG 4 by helping educators identify students who may need academic support **earlier**, enabling timely, targeted intervention while keeping teachers responsible for all decisions.


## Note:
without the ibm cloud watsonx.ai credentials are not set means the streamlit can be opens in a web browser in a demo mode.
