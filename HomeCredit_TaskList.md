# 🏦 Home Credit Default Risk — Complete Task List

> **Project Goal:** Build a production-ready credit risk scoring system with AUC 0.787+ (top 20% Kaggle), deployed as a live web app with SHAP explainability, fairness analysis, and model monitoring.
>
> **Total Estimated Time:** 3–4 weeks (4 hrs/day)
> **Stack:** Python · LightGBM · XGBoost · SHAP · FastAPI · Streamlit · Docker · Render.com · MLflow

---

## ✅ PHASE 0 — Setup & Environment

### 0.1 Install VS Code
- [x] Download and install VS Code from `code.visualstudio.com`
- [x] Install extension: **Python** (Microsoft)
- [x] Install extension: **Jupyter**
- [x] Install extension: **Pylance**
- [x] Install extension: **GitLens**
- [x] Install extension: **Rainbow CSV**
- [x] Install extension: **Docker**
- [x] Install extension: **Thunder Client**
- [x] Install extension: **indent-rainbow**

### 0.2 Create Conda Environment
- [x] Run `conda create -n home_credit python=3.11 -y`
- [x] Run `conda activate home_credit`
- [x] Install core libraries: `pandas==2.2.0 numpy==1.26.4 scikit-learn==1.4.0`
- [x] Install ML libraries: `lightgbm==4.3.0 xgboost==2.0.3 optuna==3.6.1`
- [x] Install explainability: `shap==0.45.0`
- [x] Install visualization: `matplotlib==3.8.3 seaborn==0.13.2 plotly==5.19.0`
- [x] Install API/app: `fastapi==0.110.0 uvicorn==0.29.0 pydantic==2.6.4 streamlit==1.32.0`
- [x] Install tracking: `mlflow==2.11.1 joblib==1.3.2`
- [x] Install utilities: `kaggle==1.6.6 pyarrow==15.0.0 ydata-profiling==4.7.0 python-dotenv==1.0.1`
- [x] Install ML extras: `imbalanced-learn==0.12.0 scipy==1.12.0`
- [x] Install testing: `pytest==8.1.1 httpx==0.27.0`
- [x] Select `home_credit` interpreter in VS Code (bottom right corner)

### 0.3 Kaggle API Setup
- [x] Go to `kaggle.com → Account → API → Create New Token` → download `kaggle.json`
- [x] Create `~/.kaggle/` folder and move `kaggle.json` there
- [x] Run `chmod 600 ~/.kaggle/kaggle.json` (Mac/Linux only)
- [x] Go to Kaggle competition page and **accept competition rules** (required before download)
- [x] Run `kaggle competitions download -c home-credit-default-risk`
- [x] Unzip into `data/raw/`
- [x] Verify all 8 CSV files are present
- [x] **Read** `HomeCredit_columns_description.csv` fully before writing any code

### 0.4 Create Folder Structure
- [x] Run folder creation command to create all directories at once
- [x] Create notebook files: `01_eda.ipynb`, `02_feature_engineering.ipynb`, `03_modeling.ipynb`, `04_explainability.ipynb`
- [x] Create source files: `src/__init__.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`
- [x] Create app files: `app/main.py`, `app/api.py`, all 3 pages, utils files
- [x] Create config files: `.env`, `.gitignore`, `Dockerfile`, `render.yaml`, `requirements.txt`, `README.md`

### 0.5 Git & Environment Config
- [x] Run `git init` in project root
- [x] Write `.gitignore` (exclude `data/raw/`, `data/processed/`, `*.pkl`, `*.parquet`, `.env`)
- [x] Write `.env` with `MLFLOW_TRACKING_URI`, `MODEL_PATH`, `THRESHOLD`, `API_KEY`
- [x] Make first commit: `git commit -m "chore: project setup"`
- [ ] Create GitHub repository and push

---

## ✅ PHASE 1 — Complete EDA (notebooks/01_eda.ipynb)

### 1.1 Load Data and First Inspection
- [ ] Import all libraries (pandas, numpy, matplotlib, seaborn, warnings)
- [ ] Load `application_train.csv` and `application_test.csv`
- [ ] Print train shape (expect 307,511 × 122)
- [ ] Print test shape (expect 48,744 × 121)
- [ ] Confirm only `TARGET` column differs between train and test
- [ ] Print TARGET distribution (expect 91.9% non-default, 8.1% default)
- [ ] Print dtype breakdown (float64: 65, int64: 41, object: 16)
- [ ] Print memory usage in MB

### 1.2 Missing Value Analysis
- [ ] Write `analyze_missing()` function
- [ ] Run on `application_train` and print top 30 missing columns
- [ ] Build and save missing value heatmap plot to `reports/missing_heatmap.png`
- [ ] Identify columns with >60% missing — mark for dropping
- [ ] Write markdown explanation for WHY key columns are missing (EXT_SOURCE_1, OWN_CAR_AGE, OCCUPATION_TYPE)

### 1.3 Target Analysis — Who Defaults?
- [ ] Calculate default rate for each categorical column (12 columns)
- [ ] Plot default rate bar charts for all categorical columns (4×3 grid)
- [ ] Save plot to `reports/default_by_category.png`
- [ ] Write findings: male vs female default rate
- [ ] Write findings: default rate by education type
- [ ] Write findings: default rate by income type
- [ ] Write markdown cell summarizing top 3 business insights

### 1.4 Numeric Feature Distributions
- [ ] Plot overlapping histograms (default vs non-default) for 12 key numeric features
- [ ] Save plot to `reports/numeric_distributions.png`
- [ ] **Discover DAYS_EMPLOYED=365243 anomaly** — print count and default rate for these rows
- [ ] Discover AMT_INCOME_TOTAL outliers (values >1M) — print count and range
- [ ] Write markdown cell documenting both anomalies and how to handle them

### 1.5 Correlation Analysis
- [ ] Calculate correlations of all numeric features with TARGET
- [ ] Print top 15 negative correlations (protective factors)
- [ ] Print top 15 positive correlations (risk factors)
- [ ] Plot dual horizontal bar chart — save to `reports/correlations_with_target.png`
- [ ] Build KDE plots for EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 by TARGET
- [ ] Save EXT_SOURCE plot to `reports/ext_source_distributions.png`
- [ ] Write markdown: EXT_SOURCE_2 is the single strongest predictor

### 1.6 Automated Profile Report
- [ ] Run `ydata_profiling` on 10,000 sample rows
- [ ] Save HTML report to `reports/eda_profile.html`
- [ ] Open in browser and study: correlations, missing patterns, distributions
- [ ] Write 5-bullet EDA summary markdown cell at the end of notebook

---

## ✅ PHASE 2 — Complete Feature Engineering (src/features.py)

### 2.1 Application Table Features
- [ ] Create `engineer_application_features(df)` function in `src/features.py`
- [ ] Fix `DAYS_EMPLOYED=365243` anomaly — create flag + replace with NaN
- [ ] Create age features: `AGE_YEARS`, `EMPLOYED_YEARS`, `ID_PUBLISH_YEARS`, `REGISTRATION_YEARS`, `LAST_PHONE_YEARS`
- [ ] Create ratio features: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_GOODS_RATIO`, `ANNUITY_CREDIT_RATIO`, `INCOME_PER_PERSON`, `GOODS_CREDIT_DIFF`, `CHILDREN_RATIO`, `EMPLOYED_TO_AGE_RATIO`
- [ ] Create EXT_SOURCE features: `EXT_SOURCE_MEAN`, `MAX`, `MIN`, `STD`, `PROD`, `RANGE`, `NANCOUNT`, `WEIGHTED`
- [ ] Count FLAG_DOCUMENT columns → `TOTAL_DOCS_SUBMITTED`
- [ ] Count contact flag columns → `TOTAL_CONTACTS`
- [ ] Create `SOCIAL_CIRCLE_DEF_RATE`
- [ ] Cap `AMT_INCOME_TOTAL` at 99th percentile
- [ ] Create `IS_WEEKEND_APPLY` and `IS_NIGHT_APPLY` flags
- [ ] Fill apartment/building features with median
- [ ] Apply function to both `app_train` and `app_test`
- [ ] Print shape after (expect ~155 columns)

### 2.2 Bureau Data Aggregation
- [ ] Write `aggregate_bureau()` function
- [ ] Aggregate `bureau_balance` → bureau level (months count, status flags, DPD count/max)
- [ ] Merge bureau_balance aggregates back to bureau
- [ ] Aggregate bureau → SK_ID_CURR level (loan count, active/closed/bad debt counts, amount features, DPD features, credit duration features)
- [ ] Create derived ratios: `B_DEBT_CREDIT_RATIO`, `B_OVERDUE_LOAN_RATIO`, `B_ACTIVE_LOAN_RATIO`
- [ ] Merge bureau_agg to app_train and app_test
- [ ] Print shape after merge

### 2.3 Previous Applications Aggregation
- [ ] Write `aggregate_previous_applications()` function
- [ ] Fix `AMT_APPLICATION=0` anomaly — replace with NaN
- [ ] Create `PREV_CREDIT_APP_RATIO` per loan
- [ ] Aggregate by SK_ID_CURR (count, approved/refused/canceled counts, amount features, days features, loan type counts)
- [ ] Create derived: `P_APPROVAL_RATE`, `P_REFUSAL_RATE`
- [ ] Build approved-only aggregation separately and merge
- [ ] Merge prev_agg to train and test
- [ ] Print shape after merge

### 2.4 Installments Payments Aggregation
- [ ] Write `aggregate_installments()` function
- [ ] Calculate per-payment features: `PAYMENT_DIFF`, `PAYMENT_RATIO`, `DPD`, `DBD`, `PAID_LATE`, `PAID_EARLY`, `UNDERPAID`, `OVERPAID`
- [ ] Aggregate by SK_ID_CURR (count, DPD max/mean/sum, DBD max/mean, payment diff stats, late/early ratios, underpaid stats, amount stats)
- [ ] Create recent 12-month slice and aggregate separately (4 features)
- [ ] Merge both aggregations → merge to train and test
- [ ] Print shape after merge

### 2.5 POS_CASH + Credit Card Aggregation
- [ ] Write `aggregate_pos_cash()` function
- [ ] Aggregate by SK_ID_CURR (unique loan count, months count, DPD features, status counts, instalment features)
- [ ] Create `POS_DPD_RATIO`
- [ ] Write `aggregate_credit_card()` function
- [ ] Create per-month features: `CC_UTILIZATION`, `CC_PAYMENT_RATIO`, `CC_MIN_PAYMENT_RATIO`
- [ ] Aggregate by SK_ID_CURR (card count, months, balance, credit limit, DPD, drawing features, utilization stats, payment ratios)
- [ ] Create `CC_OVER_LIMIT_COUNT`
- [ ] Merge both to train and test
- [ ] Print final feature count (expect 260+)

### 2.6 Save Processed Features
- [ ] Save `app_train` to `data/processed/train_features.parquet`
- [ ] Save `app_test` to `data/processed/test_features.parquet`
- [ ] Save feature names list to `models/feature_names.json`
- [ ] Verify: load parquet files and confirm shapes match

---

## ✅ PHASE 3 — Preprocessing Pipeline (src/train.py)

- [ ] Load train and test from parquet
- [ ] Identify categorical vs numerical features
- [ ] Convert categorical columns to `category` dtype for LightGBM
- [ ] Create X, y, X_test (drop TARGET and SK_ID_CURR)
- [ ] Assert column order matches between X and X_test
- [ ] Print final X shape, y distribution, X_test shape

---

## ✅ PHASE 4 — LightGBM Modeling

### 4.1 LightGBM 5-Fold Cross Validation
- [ ] Define `lgb_params` dictionary (all parameters including `scale_pos_weight=11`)
- [ ] Initialize `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- [ ] Initialize `oof_preds`, `test_preds`, `feature_imp`, `fold_scores`, `models` arrays
- [ ] Write training loop for 5 folds
- [ ] Within each fold: train LGBMClassifier with early stopping
- [ ] Calculate and print fold AUC
- [ ] Save test predictions as average across folds
- [ ] Collect feature importance per fold
- [ ] Save each fold model to `models/lgbm_fold{n}.pkl`
- [ ] Print OOF AUC, all fold AUCs, and std
- [ ] Save best fold model to `models/lgbm_final.pkl`

### 4.2 Feature Importance Analysis
- [ ] Average feature importance across 5 folds
- [ ] Print top 30 most important features
- [ ] Plot horizontal bar chart of top 30 — save to `reports/feature_importance_lgbm.png`
- [ ] Add mean importance reference line to plot
- [ ] Identify low importance features (<10) — list them
- [ ] Save feature importance CSV to `reports/feature_importance.csv`

---

## ✅ PHASE 5 — XGBoost + Ensemble

### 5.1 XGBoost Training
- [ ] Copy X and X_test for XGBoost
- [ ] Encode all categorical columns with `LabelEncoder` (fit on combined train+test)
- [ ] Fill NaN values with -999
- [ ] Define `xgb_params` dictionary
- [ ] Initialize OOF and test prediction arrays for XGBoost
- [ ] Write XGBoost 5-fold training loop with early stopping
- [ ] Print XGB OOF AUC and per-fold AUCs
- [ ] Save best XGB model to `models/xgb_final.pkl`

### 5.2 Ensemble Blending
- [ ] Use `scipy.optimize.minimize_scalar` to find optimal blend weight
- [ ] Calculate final ensemble OOF predictions
- [ ] Calculate final ensemble test predictions
- [ ] Print comparison: LGB AUC vs XGB AUC vs Ensemble AUC
- [ ] Save all OOF and test prediction arrays with `np.save`

---

## ✅ PHASE 6 — Hyperparameter Tuning

- [ ] Define `objective(trial)` function with 10 Optuna parameters
- [ ] Use 3-fold CV inside objective for speed
- [ ] Create Optuna study with SQLite storage (`sqlite:///models/optuna_study.db`)
- [ ] Run `study.optimize(objective, n_trials=100)` — **run overnight**
- [ ] Print best trial AUC and best parameters
- [ ] Retrain full 5-fold CV with best parameters (same loop as Phase 4)
- [ ] Compare new AUC vs original AUC
- [ ] Save best params to `models/best_lgb_params.json`

---

## ✅ PHASE 7 — Complete Model Evaluation

### 7.1 Banking-Specific Metrics
- [ ] Calculate AUC-ROC using ensemble OOF predictions
- [ ] Calculate Gini coefficient (`2 * AUC - 1`)
- [ ] Calculate KS statistic from ROC curve
- [ ] Find KS-optimal threshold
- [ ] Build 3-panel evaluation plot (ROC curve, KS plot, score distribution)
- [ ] Save to `reports/model_evaluation.png`
- [ ] Print benchmarks: AUC, Gini, KS stat

### 7.2 Threshold Optimization
- [ ] Calculate precision-recall curve
- [ ] Find F1-optimal threshold
- [ ] Run business cost analysis (FN costs 5× FP)
- [ ] Find business-optimal threshold
- [ ] Print classification report at business threshold
- [ ] Save optimal threshold to `models/optimal_threshold.txt`
- [ ] Build calibration curve — save to `reports/calibration_curve.png`
- [ ] Write markdown: interpretation of calibration quality

---

## ✅ PHASE 8 — SHAP Explainability

- [ ] Load `lgbm_final.pkl`
- [ ] Sample 2,000 rows for SHAP analysis
- [ ] Create `shap.TreeExplainer`
- [ ] Calculate SHAP values for sample
- [ ] Build global summary bar plot — save to `reports/shap_summary_bar.png`
- [ ] Build beeswarm plot — save to `reports/shap_beeswarm.png`
- [ ] Write markdown: what RED vs BLUE means on beeswarm
- [ ] Pick highest-risk applicant from OOF predictions
- [ ] Build single-applicant waterfall plot — save to `reports/shap_single_applicant.png`
- [ ] Build dependence plots for top 3 features (EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH) — save plot
- [ ] Write `explain_prediction(applicant_df, top_n=5)` helper function
- [ ] Save `shap.TreeExplainer` to `models/shap_explainer.pkl`

---

## ✅ PHASE 9 — MLflow Experiment Tracking

- [ ] Start MLflow server: `mlflow ui --port 5000`
- [ ] Set tracking URI and experiment name
- [ ] Log all `lgb_params` as parameters
- [ ] Log n_folds, n_features, n_samples, positive_rate
- [ ] Log per-fold AUC as metric with step
- [ ] Log summary metrics: oof_auc, gini, ks_stat, std_auc, ensemble_auc
- [ ] Log all report artifacts (evaluation plot, feature importance, SHAP plot)
- [ ] Log model with `mlflow.sklearn.log_model`
- [ ] Register model in Model Registry as `home_credit_lgbm`
- [ ] Open `http://localhost:5000` and verify run appears
- [ ] Log XGBoost run separately with same structure
- [ ] Compare both runs in MLflow UI

---

## ✅ PHASE 10 — Kaggle Submission

- [ ] Load test parquet and ensemble test predictions
- [ ] Create submission DataFrame (SK_ID_CURR, TARGET)
- [ ] Verify submission shape (48,744 × 2)
- [ ] Verify score range is [0, 1]
- [ ] Save to `data/submissions/lgbm_xgb_ensemble_v1.csv`
- [ ] Submit via Kaggle CLI with descriptive message
- [ ] Record public leaderboard score
- [ ] If AUC < 0.77: review feature engineering steps
- [ ] If AUC 0.77–0.78: add more aggregations from Phase 2
- [ ] If AUC > 0.78: proceed to production deployment

---

## ✅ PHASE 11 — FastAPI Complete

### 11.1 Build app/api.py
- [ ] Set up logging to both file and stdout
- [ ] Initialize FastAPI app with title, description, version
- [ ] Add CORS middleware
- [ ] Implement startup event to load model, explainer, feature_names, threshold
- [ ] Implement `HTTPBearer` security with API key from `.env`
- [ ] Write `ApplicantData` Pydantic model with all fields, validators, and defaults
- [ ] Write `RiskFactor`, `PredictionResponse`, `BatchRequest` schemas
- [ ] Write `prepare_features()` helper
- [ ] Write `score_to_tier()` helper
- [ ] Write `prob_to_score()` helper (0–1000 CIBIL-style score)
- [ ] Implement `GET /` root endpoint
- [ ] Implement `GET /health` endpoint
- [ ] Implement `POST /predict` async endpoint with SHAP explanation + background logging
- [ ] Implement `POST /predict/batch` endpoint (max 1000 applicants)
- [ ] Create `logs/` directory
- [ ] Test with Thunder Client: `/health`, `/predict`, `/predict/batch`, unauthorized request

### 11.2 Test FastAPI Locally
- [ ] Run `uvicorn app.api:app --reload --host 0.0.0.0 --port 8000`
- [ ] Open `http://localhost:8000/docs`
- [ ] Test `/health` — verify status: healthy
- [ ] Test `/predict` with valid applicant — verify response structure
- [ ] Test `/predict` with invalid income (negative) — verify 422 error
- [ ] Test `/predict` without auth header — verify 403 error
- [ ] Test `/predict/batch` with 2 applicants — verify count=2
- [ ] Check `logs/api.log` — verify predictions are being logged

---

## ✅ PHASE 12 — Streamlit App Complete

### 12.1 app/main.py Homepage
- [ ] Set page config (title, icon, layout=wide)
- [ ] Add main title and description
- [ ] Add 4 KPI metric cards (AUC, Gini, KS, Training Size)
- [ ] Add navigation description for all 3 pages

### 12.2 pages/1_Single_Prediction.py
- [ ] Load model, explainer, feature_names, threshold with `@st.cache_resource`
- [ ] Build 3-column input form (financial, personal, credit history)
- [ ] Add form submit button
- [ ] On submission: build input dict, engineer features, fill missing columns
- [ ] Run model prediction → get probability and risk score
- [ ] Display 4 result metrics (probability, credit score, decision, risk tier)
- [ ] Build and display Plotly gauge chart
- [ ] Calculate SHAP values for prediction
- [ ] Build and display risk factors dataframe
- [ ] Add caption explaining risk impact direction

### 12.3 pages/2_Batch_Scoring.py
- [ ] Load model with `@st.cache_resource`
- [ ] Add file uploader (CSV only)
- [ ] On file upload: display loaded applicant count and preview
- [ ] Add "Score All Applicants" button
- [ ] On button click: engineer features, predict, assign decisions and risk tiers
- [ ] Display 4 summary metrics (approved/review/rejected counts, avg credit score)
- [ ] Display probability distribution histogram by risk tier
- [ ] Display decision distribution pie chart
- [ ] Display top 100 scored results table
- [ ] Add CSV download button for full results

### 12.4 pages/3_Portfolio_Analytics.py
- [ ] Load OOF predictions and assign risk tiers with `@st.cache_data`
- [ ] Build 3-tab layout (Model Performance, Feature Analysis, Risk Segmentation)
- [ ] Tab 1: 4 metric cards + score distribution overlay histogram + calibration plot
- [ ] Tab 2: Feature importance horizontal bar chart (top 30)
- [ ] Tab 3: Risk segment summary table + default rate bar chart + portfolio pie chart

### 12.5 Test Streamlit App
- [ ] Run `streamlit run app/main.py`
- [ ] Test homepage loads with correct KPI values
- [ ] Test single prediction: submit form → gauge appears, SHAP table appears
- [ ] Test batch scoring: upload sample CSV → results appear, download works
- [ ] Test portfolio analytics: all 3 tabs render without errors
- [ ] Test all pages load in < 5 seconds

---

## ✅ PHASE 13 — Docker + Render Deployment

### 13.1 Dockerfile
- [ ] Write Dockerfile (python:3.11-slim base)
- [ ] Add system dependencies (build-essential, curl, git)
- [ ] Copy requirements.txt and install (layer caching)
- [ ] Copy src/, app/, models/, reports/
- [ ] Create logs/ directory
- [ ] Set EXPOSE 8501
- [ ] Add HEALTHCHECK
- [ ] Write CMD for Streamlit

### 13.2 Build and Test Locally
- [ ] Create `.dockerignore` (exclude .env, __pycache__, .git, data/raw, data/processed)
- [ ] Run `docker build -t home-credit-risk:v1 .`
- [ ] Wait for build to complete (8–12 minutes first time)
- [ ] Run `docker run -p 8501:8501 --env-file .env home-credit-risk:v1`
- [ ] Open `localhost:8501` — verify all pages work inside container
- [ ] Test single prediction inside Docker
- [ ] Run `docker ps` — verify STATUS shows healthy
- [ ] Check `docker images` — note image size

### 13.3 Deploy to Render.com
- [ ] Write `render.yaml` with web service config
- [ ] Push all code to GitHub (excluding secrets)
- [ ] Create account at render.com
- [ ] New Web Service → Connect GitHub repo
- [ ] Add `API_KEY` environment variable in Render dashboard
- [ ] Click Deploy — watch build logs
- [ ] Wait for deployment (15–20 minutes first time)
- [ ] Open live URL in incognito browser
- [ ] Test all 3 pages live
- [ ] Test single prediction in production
- [ ] Record live URL for resume and LinkedIn

---

## ✅ PHASE 14 — Tests + Documentation

### 14.1 Unit Tests (tests/test_features.py)
- [ ] Write `sample_applicant` pytest fixture
- [ ] Write `test_credit_income_ratio` — verify calculation
- [ ] Write `test_ext_source_mean` — verify average calculation
- [ ] Write `test_age_years` — verify range (30–40 for 12000 days birth)
- [ ] Write `test_days_employed_anomaly` — verify 365243 flag and NaN replacement
- [ ] Write `test_output_columns_exist` — verify required columns present
- [ ] Write `test_no_infinite_values` — verify no inf in output
- [ ] Run `pytest tests/test_features.py -v` — all tests pass

### 14.2 API Tests (tests/test_api.py)
- [ ] Write `test_health_endpoint` — status 200 and healthy
- [ ] Write `test_predict_returns_200` — with valid input
- [ ] Write `test_predict_response_structure` — all required keys present
- [ ] Write `test_probability_range` — between 0.0 and 1.0
- [ ] Write `test_decision_values` — APPROVE/REVIEW/REJECT only
- [ ] Write `test_unauthorized_returns_403` — wrong API key
- [ ] Write `test_invalid_income_rejected` — negative income returns 422
- [ ] Write `test_batch_predict` — count=2 for 2 applicants
- [ ] Run `pytest tests/test_api.py -v` — all tests pass
- [ ] Run `pytest tests/ --cov=src --cov-report=term-missing` — check coverage

### 14.3 Documentation
- [ ] Write `README.md` with live demo URL at very top
- [ ] Add project description and problem statement
- [ ] Add architecture diagram (draw.io or ASCII)
- [ ] Add model performance table (AUC, Gini, KS)
- [ ] Add tech stack table with links
- [ ] Add local setup instructions (5 commands to run)
- [ ] Add API documentation with example curl commands
- [ ] Add screenshots of Streamlit app (all 3 pages)
- [ ] Add Tableau/deployment links
- [ ] Write `MODELS.md` with 1 section per model (purpose, features, algorithm, metrics, limitations)
- [ ] Add MIT license file

---

## 🚀 EXCEPTIONAL LAYERS (Beyond Top 20%)

### Layer 1 — Model Stacking
- [ ] Train CatBoost as third base model (same 5-fold CV)
- [ ] Stack OOF predictions: LGB + XGB + CatBoost as meta-features
- [ ] Train `LogisticRegression(C=0.1)` as meta-model on stacked OOF
- [ ] Evaluate stacked AUC (expect +0.003–0.006 over best single model)
- [ ] Include stacked predictions in final submission

### Layer 2 — Neural Network (TabNet)
- [ ] `pip install pytorch-tabnet`
- [ ] Encode categoricals as integers, fill NaN with -1
- [ ] Train `TabNetClassifier` with 5-fold CV
- [ ] Compare TabNet AUC vs LGB AUC
- [ ] Add TabNet OOF to stacking meta-features

### Layer 3 — Fairness Analysis ⭐ (Biggest Differentiator)
- [ ] Merge OOF predictions back to training data
- [ ] Calculate AUC separately by `CODE_GENDER` (M/F)
- [ ] Calculate approval rate by gender
- [ ] Calculate Disparate Impact ratio (threshold: 0.80)
- [ ] Calculate Equal Opportunity Difference (threshold: <0.05)
- [ ] Build `AGE_GROUP` buckets (18–25, 26–35, 36–45, 46–55, 55+)
- [ ] Calculate AUC, approval rate, default rate per age group
- [ ] Plot 3-panel fairness charts — save to `reports/fairness_analysis.png`
- [ ] Save `reports/fairness_report.json` with all metrics and verdict
- [ ] Add Fairness tab to Streamlit Portfolio Analytics page
- [ ] Show compliance verdict: PASS / REVIEW REQUIRED

### Layer 4 — Model Monitoring with Evidently AI
- [ ] `pip install evidently`
- [ ] Split training data: first 70% as reference, last 30% as current batch
- [ ] Run `DataDriftPreset` report — save HTML
- [ ] Run `DataDriftTestPreset` — count drifted features
- [ ] Write `calculate_psi()` function (PSI thresholds: <0.1 stable, 0.1–0.2 monitor, >0.2 retrain)
- [ ] Calculate PSI for top 5 features
- [ ] Print PSI status per feature (STABLE/MONITOR/RETRAIN)
- [ ] Add Model Health tab to Streamlit showing PSI table with color coding

### Layer 5 — Counterfactual Explanations (DiCE)
- [ ] `pip install dice-ml`
- [ ] Initialize `dice_ml.Data` and `dice_ml.Model`
- [ ] Create `Dice` explainer
- [ ] Pick a rejected applicant with TARGET=0 (false positive)
- [ ] Generate 3 counterfactuals varying key features
- [ ] Display as DataFrame showing only changed features
- [ ] Add **"What would change this decision?"** button in Single Prediction page
- [ ] Show counterfactual table on REJECT results: "If X improves, decision flips to APPROVE"

### Layer 6 — Technical Blog Post
- [ ] Create Medium or Dev.to account
- [ ] Write 1,500-word post covering:
  - [ ] Why this problem is hard (imbalance, 7 tables, regulatory constraints)
  - [ ] Most surprising EDA finding (DAYS_EMPLOYED anomaly or EXT_SOURCE insight)
  - [ ] What SHAP revealed about credit risk factors
  - [ ] Fairness analysis findings
  - [ ] One thing you'd do differently (temporal validation)
- [ ] Include architecture diagram in post
- [ ] Include SHAP plots (with permission)
- [ ] Publish and share on LinkedIn with live app link
- [ ] Tag: #MachineLearning #CreditRisk #DataScience #Python

---

## 📋 Final Portfolio Launch Checklist

- [ ] Live app URL works in incognito browser
- [ ] All 3 Streamlit pages load without errors
- [ ] Single prediction: gauge + SHAP table appears
- [ ] Batch scoring: upload CSV → download results works
- [ ] Portfolio analytics: all charts render
- [ ] FastAPI `/docs` accessible at live URL
- [ ] GitHub repo is public with complete README
- [ ] Live URL at very top of README
- [ ] All model performance metrics documented in MODELS.md
- [ ] Fairness report in `reports/` folder
- [ ] Executive summary or blog post linked in README
- [ ] All unit and API tests passing
- [ ] LinkedIn post published with live URL and 3 key insights
- [ ] Resume updated with live URL and model AUC score
- [ ] Apply to 5+ companies with live URL in application

---

## 📊 Resume Line (Copy This After Completion)

```
Home Credit Default Risk Scoring System
• Production credit risk platform on 307K loan applications (7 relational tables, 250+ engineered features)
• LightGBM + XGBoost + CatBoost stacked ensemble — AUC 0.787+ (top 20% Kaggle leaderboard)
• SHAP explainability (regulatory compliance), fairness analysis (disparate impact ratio), Evidently AI monitoring
• FastAPI scoring endpoint (<100ms latency), Streamlit dashboard (single/batch/portfolio), counterfactual explanations
• Deployed via Docker on Render.com — Live: https://home-credit-risk.onrender.com
• Stack: Python, LightGBM, XGBoost, SHAP, MLflow, FastAPI, Streamlit, Docker, Optuna
```

---

*Last updated: March 2026 | Kumar Sarthak | Gurugram*
