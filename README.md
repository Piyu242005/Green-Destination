# 🌍 Green Destinations Employee Attrition Analysis

### Explainable Employee Attrition Risk Prediction System

An end-to-end HR analytics and machine-learning system for employee attrition analysis, visual exploration, explainable risk prediction, and API-based serving.

> **Purpose:** This project demonstrates the complete journey from HR data exploration and factor analysis to machine-learning deployment.

## 🖼️ Screenshot Gallery

### 1. Load and Explore the Data

<p align="center"><img src="./screenshots/1.%20Load%20and%20Explore%20the%20Data.png" alt="Load and Explore the Data" width="900"></p>

### 2. Calculate Attrition Rate

<p align="center"><img src="./screenshots/2.%20Calculate%20Attrition%20Rate.png" alt="Calculate Attrition Rate" width="900"></p>

### 3. Factor Analysis — Age vs Attrition

<p align="center"><img src="./screenshots/3.%20Factor%20Analysis%20Age%20vs%20Attrition.png" alt="Age vs Attrition analysis" width="900"></p>

### 4. Factor Analysis — Years at Company vs Attrition

<p align="center"><img src="./screenshots/4.%20Factor%20Analysis%20Years%20at%20Company%20vs%20Attrition.png" alt="Years at Company vs Attrition analysis" width="900"></p>

### 5. Factor Analysis — Monthly Income vs Attrition

<p align="center"><img src="./screenshots/5.%20Factor%20Analysis%20Monthly%20Income%20vs%20Attrition.png" alt="Monthly Income vs Attrition analysis" width="900"></p>

### 6. Summary & Key Findings

<p align="center"><img src="./screenshots/6.%20Summary%20%26%20Key%20Findings.png" alt="Summary and Key Findings" width="900"></p>

### 7. Summary & Key Findings — Alternate View

<p align="center"><img src="./screenshots/6.%20Summary%20%26%20Key%20Findings%20%282%29.png" alt="Summary and Key Findings alternate view" width="900"></p>

### 🚀 FastAPI Backend

<p align="center"><img src="./screenshots/FastAPI%20backend.png" alt="FastAPI backend" width="900"></p>

### 🖥️ Streamlit Dashboard

<p align="center"><img src="./screenshots/Streamlit%20dashboard.png" alt="Streamlit dashboard" width="900"></p>

### 📌 Project Preview

<p align="center"><img src="./screenshots/github.jpeg" alt="Project preview" width="900"></p>

## 🎯 Problem

Employee turnover creates recruitment, training and productivity costs. The project analyzes attrition patterns and provides an ML-based decision-support workflow for identifying higher-risk employees.

## 🏗️ Architecture

```mermaid
graph LR
    DATA[HR Dataset] --> EDA[EDA & Factor Analysis]
    EDA --> FE[Feature Engineering]
    FE --> ML[Leakage-safe ML Pipeline]
    ML --> EVAL[Model Evaluation]
    EVAL --> API[FastAPI]
    EVAL --> UI[Streamlit]
    ML --> XAI[SHAP Explainability]
    API --> ACTION[Retention Decision Support]
```

## ✨ Engineering Highlights

- HR data exploration and attrition-rate analysis
- Factor analysis across age, tenure and income
- Leakage-safe preprocessing with `ImbPipeline`
- One-hot encoding and SMOTE
- GridSearchCV with stratified cross-validation
- Validation-based **F2 threshold optimization**
- Untouched final test set for evaluation
- SHAP individual explanations
- Model comparison benchmarking
- Data/prediction drift monitoring utilities
- Fairness and representation analysis
- Pydantic API validation
- `/health` model-health endpoint
- HR analytics dashboard
- Responsible-AI model card

## 📊 Evaluation

**ROC-AUC · PR-AUC · Recall · Precision · F1 · F2 · Confusion Matrix**

Generate current model metrics with:

```bash
python src/train.py
```

## 🔌 API

```text
POST /predict
GET  /health
GET  /docs
```

## 🖥️ Applications

```bash
streamlit run app/main.py
streamlit run app/analytics_dashboard.py
uvicorn app/advanced_api:app --reload --port 8000
```

## 🚀 Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python src/train.py
pytest -q
```

## 📁 Structure

```text
data/        # HR dataset
models/      # trained artifacts + metadata
src/         # feature engineering, training, comparison, monitoring
app/         # FastAPI + Streamlit applications
screenshots/ # Project screenshot gallery
tests/       # automated tests
docs/        # production checklist
MODEL_CARD.md
Dockerfile
requirements.txt
```

## 🔐 Responsible AI

This is an **HR decision-support system**, not an automated employment decision-maker. Predictions should be reviewed by qualified HR professionals and should not independently determine hiring, termination, promotion or compensation decisions.

## 👨‍💻 Author

**Piyush Ramteke** — Data Scientist | AI/ML Engineer
