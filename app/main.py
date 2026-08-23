import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import shap
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features

st.set_page_config(page_title="Green Destinations AI", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.hero {padding: 1.2rem 1.5rem; border-radius: 16px; background: linear-gradient(135deg,#0f172a,#164e63); color:white; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.2rem;}
.hero p {margin:.35rem 0 0; opacity:.85;}
.kpi {padding:1rem; border:1px solid rgba(128,128,128,.25); border-radius:14px; background:var(--background-color); box-shadow:0 2px 10px rgba(0,0,0,.04);}
.kpi h2 {margin:.25rem 0 0;}
.risk-high {padding:1.3rem; border-radius:16px; background:rgba(239,68,68,.10); border:1px solid rgba(239,68,68,.35);}
.risk-low {padding:1.3rem; border-radius:16px; background:rgba(16,185,129,.10); border:1px solid rgba(16,185,129,.35);}
.currency-card {padding:1rem 1.1rem; border:1px solid rgba(128,128,128,.25); border-radius:14px; background:var(--background-color);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🌱 Green Destinations</h1>
<p>Employee Attrition Intelligence • Explainable ML • HR Decision Support</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    if not os.path.exists(config.MODEL_PIPELINE_PATH):
        return None, None
    model = joblib.load(config.MODEL_PIPELINE_PATH)
    background = pd.read_pickle(config.SHAP_BACKGROUND_PATH) if os.path.exists(config.SHAP_BACKGROUND_PATH) else None
    return model, background

@st.cache_data
def load_hr_data():
    path = config.DATA_PATH
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

model, background = load_assets()
df = load_hr_data()
threshold = config.load_threshold()

if model is None:
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

USD_TO_INR = 85.0

def usd_to_inr(value: float) -> float:
    return value * USD_TO_INR

def inr_to_usd(value: float) -> float:
    return value / USD_TO_INR

def money_usd(value: float) -> str:
    return f"${value:,.0f}"

def money_inr(value: float) -> str:
    return f"₹{value:,.0f}"

if not df.empty:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi"><b>Employees</b><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    attrition = (df["Attrition"] == "Yes").mean() * 100
    k2.markdown(f'<div class="kpi"><b>Attrition Rate</b><h2>{attrition:.1f}%</h2></div>', unsafe_allow_html=True)
    overtime_rate = (df["OverTime"] == "Yes").mean() * 100
    k3.markdown(f'<div class="kpi"><b>Overtime Rate</b><h2>{overtime_rate:.1f}%</h2></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><b>Retention Opportunity</b><h2>{attrition:.1f}%</h2></div>', unsafe_allow_html=True)

    avg_income_usd = float(df["MonthlyIncome"].mean())
    avg_income_inr = usd_to_inr(avg_income_usd)
    c_usd, c_inr = st.columns(2)
    with c_usd:
        st.markdown(f'<div class="currency-card"><b>💵 Average Monthly Income — USD</b><h2>{money_usd(avg_income_usd)}</h2></div>', unsafe_allow_html=True)
    with c_inr:
        st.markdown(f'<div class="currency-card"><b>🇮🇳 Average Monthly Income — INR</b><h2>{money_inr(avg_income_inr)}</h2></div>', unsafe_allow_html=True)

st.markdown("### 👤 Employee Risk Assessment")
with st.sidebar:
    st.header("Employee Profile")
    age = st.slider("Age", 18, 60, 30)
    dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])

    currency = st.radio("Income Currency", ["USD ($)", "INR (₹)"], horizontal=True)
    if currency == "USD ($)":
        income_input = st.number_input("Monthly Income (USD)", 1000, 20000, 5000, step=500)
        income_usd = float(income_input)
        st.caption(f"Equivalent INR: **{money_inr(usd_to_inr(income_usd))}**")
    else:
        income_input = st.number_input("Monthly Income (INR)", 85000, 1700000, 425000, step=5000)
        income_usd = inr_to_usd(float(income_input))
        st.caption(f"Equivalent USD: **{money_usd(income_usd)}**")

    overtime = st.selectbox("Overtime", ["Yes", "No"])
    total_years = st.slider("Total Working Years", 0, 40, 10)
    years_at_co = st.slider("Years at Company", 0, 40, 5)
    role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
    job_sat = st.slider("Job Satisfaction", 1, 4, 3)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    distance = st.number_input("Distance From Home", 0, 100, 5)
    analyze = st.button("🔍 Analyze Attrition Risk", type="primary", use_container_width=True)

if analyze:
    input_data = {
        "Age": age, "Department": dept, "MonthlyIncome": income_usd, "OverTime": overtime,
        "TotalWorkingYears": total_years, "YearsAtCompany": years_at_co, "JobRole": role,
        "EnvironmentSatisfaction": env_sat, "JobSatisfaction": job_sat, "MonthlyRate": 14313,
        "DailyRate": 802, "HourlyRate": 66, "BusinessTravel": travel, "DistanceFromHome": distance,
        "Education": 3, "EducationField": "Life Sciences", "Gender": gender, "JobInvolvement": 3,
        "JobLevel": 2, "MaritalStatus": marital, "NumCompaniesWorked": 1, "PercentSalaryHike": 15,
        "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 0,
        "TrainingTimesLastYear": 3, "WorkLifeBalance": 3, "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
    }
    input_df = engineer_features(pd.DataFrame([input_data]))
    prob = float(model.predict_proba(input_df)[0][1])
    high = prob >= threshold

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("### 🎯 Risk Score")
        box = "risk-high" if high else "risk-low"
        label = "HIGH RISK" if high else "LOW RISK"
        st.markdown(f'<div class="{box}"><h1>{prob*100:.1f}%</h1><h3>{"🔴" if high else "🟢"} {label}</h3><p>Decision threshold: {threshold:.2f}</p></div>', unsafe_allow_html=True)
        st.progress(min(prob, 1.0))
        st.markdown("### 💡 Recommended Action")
        if high:
            st.warning("Conduct a targeted retention discussion. Review workload, overtime, satisfaction, compensation and career progression.")
        else:
            st.success("Continue normal engagement and periodic satisfaction monitoring.")

        report = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(timespec="seconds"), "risk_probability": round(prob, 4),
            "risk_level": label, "threshold": threshold, "department": dept, "job_role": role,
            "monthly_income_usd": round(income_usd, 2), "monthly_income_inr": round(usd_to_inr(income_usd), 2), "overtime": overtime
        }])
        st.download_button("⬇️ Download Prediction Report", report.to_csv(index=False), "attrition_prediction.csv", "text/csv", use_container_width=True)

    with right:
        st.markdown("### 🧠 Why This Prediction?")
        try:
            transformed = model.named_steps["preprocessor"].transform(input_df)
            classifier = model.named_steps.get("classifier", model.named_steps.get("model"))
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)
            names = model.named_steps["preprocessor"].get_feature_names_out()
            if isinstance(shap_values, list): values = shap_values[1][0]
            elif getattr(shap_values, "ndim", 0) == 3: values = shap_values[0, :, 1]
            else: values = shap_values[0]
            if hasattr(values, "values"): values = values.values
            pairs = pd.DataFrame({"feature": names, "impact": values}).sort_values("impact", key=abs, ascending=False).head(10)
            st.bar_chart(pairs.set_index("feature")["impact"])
            st.caption("Positive SHAP values increase predicted attrition risk; negative values reduce it.")
        except Exception as exc:
            st.warning(f"SHAP explanation could not be rendered: {exc}")

st.divider()
st.markdown("### 📊 HR Analytics")
if not df.empty:
    f1, f2 = st.columns(2)
    with f1:
        selected_dept = st.selectbox("Department filter", ["All"] + sorted(df["Department"].unique().tolist()))
    with f2:
        selected_role = st.selectbox("Job role filter", ["All"] + sorted(df["JobRole"].unique().tolist()))
    view = df.copy()
    if selected_dept != "All": view = view[view["Department"] == selected_dept]
    if selected_role != "All": view = view[view["JobRole"] == selected_role]
    a, b = st.columns(2)
    with a:
        st.markdown("**Attrition by Department**")
        chart = pd.crosstab(view["Department"], view["Attrition"], normalize="index").get("Yes", pd.Series(dtype=float)) * 100
        st.bar_chart(chart)
    with b:
        st.markdown("**Attrition by Job Role**")
        chart = pd.crosstab(view["JobRole"], view["Attrition"], normalize="index").get("Yes", pd.Series(dtype=float)) * 100
        st.bar_chart(chart)

st.caption("Decision-support only: predictions should not independently determine hiring, termination, promotion or compensation decisions.")
