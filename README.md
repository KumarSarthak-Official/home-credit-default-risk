# 🏦 Home Credit Default Risk — Production ML Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-2B7DB4?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-EC7B3A?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?style=for-the-badge)

### 🚀 Live Demo → [home-credit-risk.onrender.com](https://home-credit-risk.onrender.com)
### 📊 API Docs → [home-credit-risk.onrender.com/docs](https://home-credit-risk.onrender.com/docs)

> **Note:** Free tier — may take 30–60 seconds to wake up on first visit.

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Live Screenshots](#-live-screenshots)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [ML Models](#-ml-models)
- [Fairness Analysis](#-fairness-analysis)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start (Local)](#-quick-start-local)
- [API Documentation](#-api-documentation)
- [Key Findings](#-key-findings-from-eda)
- [Feature Engineering](#-feature-engineering-summary)
- [Kaggle Leaderboard](#-kaggle-leaderboard)
- [Author](#-author)

---

## 🎯 Problem Statement

[Home Credit](https://www.homecredit.net/) serves 9+ countries providing consumer loans to underserved customers — people with **little or no credit history**. The challenge: how do you decide who gets a loan when there's no credit score?

**Business impact of wrong predictions:**
- Approve a defaulter → direct financial loss
- Reject a good applicant → lost revenue + unfair exclusion of underserved customers

**The data challenge:**
- 307,511 loan applications across **7 relational tables**
- **8.1% default rate** — severe class imbalance
- Missing values in 67 of 122 columns
- Hidden anomalies: `DAYS_EMPLOYED = 365243` (coded flag for pensioners)

**Goal:** Build a model that predicts default probability — accurate enough for production, explainable enough for regulatory compliance, and fair across demographic groups.

---

## 💡 Solution Overview

This project builds a **complete production ML pipeline** — from raw data to deployed web application:

```
Raw Data (7 CSV files, ~16M rows total)
    ↓ Feature Engineering (250+ features)
    ↓ LightGBM + XGBoost + CatBoost ensemble
    ↓ SHAP Explainability + Fairness Analysis
    ↓ FastAPI scoring endpoint
    ↓ Streamlit dashboard
    ↓ Docker + Render.com deployment
         → Live URL accessible to anyone
```

**What makes this different from a notebook:**
- ✅ Live deployed web app — not just a Jupyter notebook
- ✅ SHAP explanations per applicant — regulatory compliance
- ✅ Fairness analysis across gender and age groups
- ✅ Model drift monitoring with Evidently AI
- ✅ Counterfactual explanations — "what would flip this rejection?"
- ✅ Production API with auth, validation, logging, and async
- ✅ MLflow experiment tracking across all model runs
- ✅ Full unit and integration test coverage

---

## 📸 Live Screenshots

| Single Prediction | Batch Scoring |
|---|---|
| ![Single Prediction](reports/screenshots/single_prediction.png) | ![Batch Scoring](reports/screenshots/batch_scoring.png) |

| Portfolio Analytics | SHAP Explanation |
|---|---|
| ![Portfolio](reports/screenshots/portfolio_analytics.png) | ![SHAP](reports/screenshots/shap_explanation.png) |

| Fairness Dashboard | Model Health |
|---|---|
| ![Fairness](reports/screenshots/fairness_dashboard.png) | ![Monitoring](reports/screenshots/model_health.png) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER / RECRUITER                      │
│           home-credit-risk.onrender.com                  │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌──────────┐   ┌──────────────┐  ┌──────────┐
   │Streamlit │   │  FastAPI     │  │ /docs    │
   │  App     │   │  REST API    │  │ Swagger  │
   │ 4 pages  │   │ /predict     │  │   UI     │
   └────┬─────┘   └──────┬───────┘  └──────────┘
        │                │
        └───────┬─────────┘
                ▼
   ┌────────────────────────┐
   │     ML Model Layer     │
   │  LightGBM (primary)    │
   │  XGBoost (ensemble)    │
   │  SHAP Explainer        │
   │  Fairness Analyzer     │
   │  Evidently Monitor     │
   └────────────┬───────────┘
                ▼
   ┌────────────────────────┐
   │   Data / Storage       │
   │  Parquet feature files │
   │  Joblib model files    │
   │  MLflow experiment DB  │
   └────────────────────────┘
                ▼
   ┌────────────────────────┐
   │   Infrastructure       │
   │  Docker Container      │
   │  Render.com (hosting)  │
   │  GitHub (CI/CD)        │
   └────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Score | Benchmark |
|--------|-------|-----------|
| **AUC-ROC** | **0.787** | Baseline: 0.740 |
| **Gini Coefficient** | **0.574** | Industry good: >0.50 |
| **KS Statistic** | **0.432** | Excellent: >0.40 |
| **Brier Score** | **0.064** | Lower is better |
| **Kaggle Leaderboard** | **Top 20%** | Out of 7,198 teams |

### Score Distribution

The model clearly separates defaulters from non-defaulters:

```
Non-Defaulters  ████████████████░░░░░░░░  concentrated at low scores
Defaulters      ░░░░░░░░░░░░████████████  concentrated at high scores
                0.0          0.5         1.0
                        Default Probability
```

### Cross-Validation Results (5-Fold)

| Fold | LightGBM AUC | XGBoost AUC |
|------|-------------|-------------|
| Fold 1 | 0.7851 | 0.7812 |
| Fold 2 | 0.7879 | 0.7841 |
| Fold 3 | 0.7863 | 0.7829 |
| Fold 4 | 0.7872 | 0.7835 |
| Fold 5 | 0.7858 | 0.7822 |
| **OOF** | **0.7865** | **0.7828** |
| **Ensemble** | **0.7874** | |

---

## 🤖 ML Models

### Model 1: LightGBM Classifier (Primary)

```python
lgb_params = {
    'objective':        'binary',
    'n_estimators':     5000,
    'learning_rate':    0.05,
    'num_leaves':       31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'scale_pos_weight': 11,    # handles 8% class imbalance
}
```

**Why LightGBM:**
- Handles NaN natively (no imputation needed)
- Native categorical support (no one-hot encoding)
- 10× faster than XGBoost on this dataset size
- Best single-model AUC on this dataset type

### Model 2: XGBoost Classifier (Ensemble member)

Adds diversity to the ensemble. Trained with identical 5-fold CV but different underlying algorithm.

### Ensemble Strategy

Optimal blend found via `scipy.optimize.minimize_scalar`:

```
Final Prediction = 0.61 × LightGBM_OOF + 0.39 × XGBoost_OOF
```

### Top 10 Most Important Features (SHAP)

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `EXT_SOURCE_2` | External credit bureau score 2 |
| 2 | `EXT_SOURCE_3` | External credit bureau score 3 |
| 3 | `EXT_SOURCE_MEAN` | Average of all 3 bureau scores |
| 4 | `DAYS_BIRTH` | Applicant age (older = less risky) |
| 5 | `CREDIT_INCOME_RATIO` | Loan amount / annual income |
| 6 | `ANNUITY_INCOME_RATIO` | Monthly payment / income |
| 7 | `I_DPD_MAX` | Max days past due on installments |
| 8 | `B_DPD_MAX` | Max days past due on bureau credits |
| 9 | `DAYS_EMPLOYED` | Employment duration |
| 10 | `EXT_SOURCE_WEIGHTED` | Weighted bureau score combination |

---

## ⚖️ Fairness Analysis

Banking AI models are subject to regulatory scrutiny. This project includes a **full fairness audit** across protected attributes.

### Gender Fairness

| Group | AUC | Approval Rate | Default Rate |
|-------|-----|---------------|--------------|
| Male | 0.786 | 72.3% | 9.8% |
| Female | 0.789 | 74.1% | 7.2% |

**Disparate Impact Ratio:** 0.976 ✅ *(threshold: ≥0.80)*

**Equal Opportunity Difference:** 0.003 ✅ *(threshold: <0.05)*

### Age Group Fairness

| Age Group | AUC | Approval Rate |
|-----------|-----|---------------|
| 18–25 | 0.771 | 63.2% |
| 26–35 | 0.783 | 71.8% |
| 36–45 | 0.792 | 74.9% |
| 46–55 | 0.795 | 76.1% |
| 55+ | 0.788 | 77.3% |

**Overall Fairness Verdict: PASS** — Model does not exhibit discriminatory patterns against protected groups.

> The 18–25 age group has a lower approval rate which reflects their genuinely higher default risk (statistically validated), not model bias.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11 | Core development |
| **ML - Primary** | LightGBM 4.3 | Main classification model |
| **ML - Ensemble** | XGBoost 2.0 | Ensemble diversity |
| **ML - Extras** | CatBoost, TabNet | Stacking layer |
| **Explainability** | SHAP 0.45 | Per-applicant explanations |
| **Fairness** | Custom + Evidently | Regulatory compliance |
| **Tuning** | Optuna 3.6 | 100-trial hyperparameter search |
| **Tracking** | MLflow 2.11 | Experiment management |
| **API** | FastAPI 0.110 | Production REST API |
| **Frontend** | Streamlit 1.32 | Interactive web dashboard |
| **Visualization** | Plotly 5.19 | Interactive charts |
| **Data** | Pandas 2.2, Parquet | Data processing + storage |
| **Testing** | Pytest 8.1 | Unit + integration tests |
| **Container** | Docker | Reproducible deployment |
| **Hosting** | Render.com | Cloud deployment |
| **CI/CD** | GitHub → Render | Auto-deploy on push |
| **Monitoring** | Evidently AI | Data drift detection |

---

## 📁 Project Structure

```
home_credit/
│
├── 📂 data/
│   ├── raw/                    ← Original Kaggle CSVs (not in git)
│   ├── processed/              ← Engineered features as parquet (not in git)
│   └── submissions/            ← Kaggle submission files
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb            ← Complete EDA (50+ analysis steps)
│   ├── 02_feature_engineering.ipynb  ← All 7 files aggregated
│   ├── 03_modeling.ipynb       ← LGB + XGB + ensemble
│   └── 04_explainability.ipynb ← SHAP + fairness analysis
│
├── 📂 src/
│   ├── __init__.py
│   ├── features.py             ← All feature engineering functions
│   ├── train.py                ← Training pipeline
│   ├── evaluate.py             ← Evaluation metrics
│   └── predict.py              ← Prediction utilities
│
├── 📂 models/
│   ├── lgbm_final.pkl          ← Best LightGBM model
│   ├── xgb_final.pkl           ← Best XGBoost model
│   ├── shap_explainer.pkl      ← SHAP TreeExplainer
│   ├── feature_names.json      ← Ordered feature list
│   ├── optimal_threshold.txt   ← Business-optimal threshold
│   └── best_lgb_params.json    ← Optuna best parameters
│
├── 📂 app/
│   ├── main.py                 ← Streamlit homepage
│   ├── api.py                  ← FastAPI application
│   └── pages/
│       ├── 1_Single_Prediction.py   ← Score one applicant
│       ├── 2_Batch_Scoring.py       ← Score CSV batch
│       ├── 3_Portfolio_Analytics.py ← Model insights
│       └── 4_ML_Predictions.py      ← Advanced ML tabs
│
├── 📂 reports/
│   ├── eda_profile.html        ← ydata-profiling report
│   ├── model_evaluation.png    ← ROC, KS, score dist
│   ├── shap_summary_bar.png    ← Global feature importance
│   ├── shap_beeswarm.png       ← Direction of effects
│   ├── fairness_analysis.png   ← Fairness charts
│   ├── fairness_report.json    ← Compliance metrics
│   └── feature_importance.csv  ← Feature importance data
│
├── 📂 tests/
│   ├── test_features.py        ← Unit tests for feature engineering
│   └── test_api.py             ← API integration tests
│
├── Dockerfile                  ← Container definition
├── render.yaml                 ← Render.com deployment config
├── requirements.txt            ← Pinned dependencies
├── .env.example                ← Environment variables template
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start (Local)

### Prerequisites
- Python 3.11+
- Anaconda or Miniconda
- Docker Desktop (optional, for container testing)
- Kaggle API key

### 1. Clone and Setup Environment

```bash
git clone https://github.com/kumarsarthak98/home-credit-default-risk.git
cd home-credit-default-risk

conda create -n home_credit python=3.11 -y
conda activate home_credit
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Place kaggle.json in ~/.kaggle/ first
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/raw/
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env — set your API_KEY
```

### 4. Run Feature Engineering + Training

```bash
# Open notebooks in order:
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb
jupyter notebook notebooks/04_explainability.ipynb
```

### 5. Start the Streamlit App

```bash
streamlit run app/main.py
# Opens at http://localhost:8501
```

### 6. Start the FastAPI Server (optional)

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
# Swagger UI at http://localhost:8000/docs
```

### 7. Run with Docker

```bash
docker build -t home-credit-risk:v1 .
docker run -p 8501:8501 --env-file .env home-credit-risk:v1
# Open http://localhost:8501
```

### 8. Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔌 API Documentation

**Base URL:** `https://home-credit-risk.onrender.com`

**Authentication:** Bearer token in Authorization header

```bash
Authorization: Bearer your-api-key
```

### Endpoints

#### `GET /health`
Returns model health status.

```bash
curl https://home-credit-risk.onrender.com/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "threshold": 0.45,
  "timestamp": "2026-03-19T09:30:00"
}
```

---

#### `POST /predict`
Score a single loan applicant.

```bash
curl -X POST https://home-credit-risk.onrender.com/predict \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 400000,
    "AMT_ANNUITY": 20000,
    "DAYS_BIRTH": -12775,
    "DAYS_EMPLOYED": -2190,
    "EXT_SOURCE_1": 0.72,
    "EXT_SOURCE_2": 0.65,
    "EXT_SOURCE_3": 0.58,
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y"
  }'
```

**Response:**

```json
{
  "default_probability": 0.1823,
  "decision": "APPROVE",
  "risk_tier": "LOW",
  "risk_score": 818,
  "top_risk_factors": [
    {
      "feature": "EXT_SOURCE_2",
      "value": 0.65,
      "impact": -0.0842,
      "direction": "DECREASES risk"
    },
    {
      "feature": "CREDIT_INCOME_RATIO",
      "value": 2.222,
      "impact": 0.0631,
      "direction": "INCREASES risk"
    }
  ],
  "model_version": "lgbm_v1.0",
  "timestamp": "2026-03-19T09:30:00"
}
```

**Decision Rules:**

| Default Probability | Decision | Risk Tier |
|--------------------|----------|-----------|
| < 0.35 | ✅ APPROVE | LOW |
| 0.35 – 0.60 | ⚠️ REVIEW | MEDIUM |
| > 0.60 | 🚫 REJECT | HIGH |

---

#### `POST /predict/batch`
Score up to 1,000 applicants in one call.

```bash
curl -X POST https://home-credit-risk.onrender.com/predict/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "applicants": [
      { "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 400000, ... },
      { "AMT_INCOME_TOTAL": 90000,  "AMT_CREDIT": 250000, ... }
    ]
  }'
```

**Response:**

```json
{
  "count": 2,
  "results": [
    { "default_probability": 0.18, "decision": "APPROVE", "risk_tier": "LOW",    "risk_score": 820 },
    { "default_probability": 0.67, "decision": "REJECT",  "risk_tier": "HIGH",   "risk_score": 330 }
  ]
}
```

---

## 🔍 Key Findings from EDA

**1. External Bureau Scores are the Strongest Predictors**
Applicants with `EXT_SOURCE_2 < 0.3` default at **3× the rate** of those with `EXT_SOURCE_2 > 0.7`. These three scores alone can achieve AUC ~0.74.

**2. The DAYS_EMPLOYED=365243 Anomaly**
55,374 applicants (18%) have `DAYS_EMPLOYED = 365243` — a placeholder value for pensioners/retired persons. This is NOT actual employment duration. Treating it as numeric instead of a flag degraded AUC by ~0.004.

**3. Credit-to-Income Ratio is Highly Predictive**
Applicants with `CREDIT_INCOME_RATIO > 4` (loan is more than 4× annual income) default at **2.4× the average rate**. This makes intuitive business sense.

**4. Age Effect is Real**
Younger applicants (18–25) default at **12.3%** vs **6.1%** for those aged 46–55. However, this reflects genuine financial stability differences, not model bias (confirmed by fairness analysis).

**5. Payment History is Gold**
`I_DPD_MAX` (max days past due on historical installments) is the #4 feature in importance. Applicants who have ever been >30 days late default at **4× the rate** of those with perfect payment history.

---

## 🔧 Feature Engineering Summary

**Total features created: 260+** across all 7 data sources.

| Source File | Features Created | Key Signals |
|-------------|-----------------|-------------|
| `application_train.csv` | 35 | Age, employment, income ratios, EXT_SOURCE combos |
| `bureau.csv` + `bureau_balance.csv` | 28 | Past loan counts, DPD history, debt ratios |
| `previous_application.csv` | 22 | Approval rates, credit amounts, refusal history |
| `installments_payments.csv` | 22 | Payment behavior, late payment ratios, DPD stats |
| `POS_CASH_balance.csv` | 12 | POS loan DPD, completion rates |
| `credit_card_balance.csv` | 16 | Utilization, payment ratios, over-limit counts |
| **Interactions** | ~125 | Cross-table ratios, weighted combinations |

---

## 🏆 Kaggle Leaderboard

| Submission | Public AUC | Description |
|------------|-----------|-------------|
| Baseline (Logistic Regression) | 0.740 | Simple model, no FE |
| LightGBM basic | 0.771 | Application table only |
| LightGBM + all tables | 0.781 | Full feature engineering |
| LGB + XGB ensemble | 0.787 | **Current best** |
| + CatBoost stacking | ~0.791 | In progress |

**Competition:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) · 7,198 teams

---

## 📈 MLflow Experiments

All training runs are tracked with MLflow. To view:

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

Tracked per run: all hyperparameters, per-fold AUC, OOF AUC, Gini, KS statistic, feature importance CSV, SHAP plots.

---

## 🧪 Test Coverage

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

```
tests/test_features.py::test_credit_income_ratio          PASSED
tests/test_features.py::test_ext_source_mean              PASSED
tests/test_features.py::test_age_years                    PASSED
tests/test_features.py::test_days_employed_anomaly        PASSED
tests/test_features.py::test_output_columns_exist         PASSED
tests/test_features.py::test_no_infinite_values           PASSED
tests/test_api.py::test_health_endpoint                   PASSED
tests/test_api.py::test_predict_returns_200               PASSED
tests/test_api.py::test_predict_response_structure        PASSED
tests/test_api.py::test_probability_range                 PASSED
tests/test_api.py::test_decision_values                   PASSED
tests/test_api.py::test_unauthorized_returns_403          PASSED
tests/test_api.py::test_invalid_income_rejected           PASSED
tests/test_api.py::test_batch_predict                     PASSED

Coverage: src/features.py: 94%  src/train.py: 87%
```

---

## 🚀 Deployment

**Production:** Render.com (free tier)
**Container:** Docker (python:3.11-slim)
**CI/CD:** GitHub → auto-deploy on push to `main`

```yaml
# render.yaml
services:
  - type: web
    name: home-credit-risk
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0
```

Every push to `main` branch triggers an automatic redeploy on Render.

---

## 📄 Resume Line

```
Home Credit Default Risk Scoring System (Live: home-credit-risk.onrender.com)
Built production credit risk platform on 307K real loan applications (7 relational
tables, 260+ engineered features). LightGBM + XGBoost ensemble — AUC 0.787 (top 20%
Kaggle). Includes SHAP explainability, fairness audit (disparate impact ratio),
Evidently AI drift monitoring, counterfactual explanations (DiCE), FastAPI endpoint
(<100ms), batch scoring, Docker deployment.
Stack: Python · LightGBM · XGBoost · SHAP · MLflow · FastAPI · Streamlit · Docker
```

---

## 👤 Author

**Kumar Sarthak**

📍 Gurugram, India
📧 kumarsarthakofficial@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/kumarsarthak98)
🐙 [GitHub](https://github.com/kumarsarthak98)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Home Credit Group](https://www.homecredit.net/) for providing the dataset
- [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk) competition community
- Top Kaggle solutions that inspired the feature engineering approach
- [SHAP](https://github.com/slundberg/shap) library by Scott Lundberg
- [Evidently AI](https://www.evidentlyai.com/) for model monitoring

---

<div align="center">

**⭐ If this project helped you, please give it a star on GitHub ⭐**

Made with ❤️ by Kumar Sarthak

</div>
