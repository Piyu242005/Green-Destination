import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features

st.set_page_config(page_title="Green Destinations AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stMetric { background-color: white; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🌱 Green Destinations: Strategic Attrition Intelligence")
st.markdown("#### Developed by: Piyush Ramteke")
st.markdown("---")


@st.cache_resource
def load_assets():
    if not os.path.exists(config.MODEL_PIPELINE_PATH):
        return None, None
    model = joblib.load(config.MODEL_PIPELINE_PATH)
    background = None
    if os.path.exists(config.SHAP_BACKGROUND_PATH):
        background = pd.read_pickle(config.SHAP_BACKGROUND_PATH)
    return model, background


model, background = load_assets()
threshold = config.load_threshold()

if model is None:
    st.error("❌ Model not found. Run `python src/train.py` first.")
    st.stop()

with st.sidebar:
    st.header("👤 Employee Profile")
    age = st.slider("Age", 18, 60, 30)
    dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
    overtime = st.selectbox("Overtime", ["Yes", "No"])
    total_years = st.slider("Total Working Years", 0, 40, 10)
    years_at_co = st.slider("Years at Company", 0, 40, 5)
    role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
    job_sat = st.slider("Job Satisfaction", 1, 4, 3)


if st.button("Analyze Attrition Risk", type="primary"):
    input_data = {
        "Age": age, "Department": dept, "MonthlyIncome": income, "OverTime": overtime,
        "TotalWorkingYears": total_years, "YearsAtCompany": years_at_co, "JobRole": role,
        "EnvironmentSatisfaction": env_sat, "JobSatisfaction": job_sat,
        "MonthlyRate": 14313, "DailyRate": 802, "HourlyRate": 66, "BusinessTravel": "Travel_Rarely",
        "DistanceFromHome": 5, "Education": 3, "EducationField": "Life Sciences", "Gender": "Male",
        "JobInvolvement": 3, "JobLevel": 2, "MaritalStatus": "Single", "NumCompaniesWorked": 1,
        "PercentSalaryHike": 15, "PerformanceRating": 3, "RelationshipSatisfaction": 3,
        "StockOptionLevel": 0, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
        "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
    }

    input_df = engineer_features(pd.DataFrame([input_data]))
    prob = float(model.predict_proba(input_df)[0][1])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Prediction")
        st.metric("Attrition Probability", f"{prob * 100:.1f}%")
        st.caption(f"Decision threshold: {threshold:.2f} (selected on validation data)")
        if prob >= threshold:
            st.error("### High Risk")
            st.write("Targeted retention strategy required.")
        else:
            st.success("### Low Risk")
            st.write("Employee is likely to stay.")

    with col2:
        st.subheader("Explainable AI (SHAP)")
        try:
            transformed_data = model.named_steps["preprocessor"].transform(input_df)
            classifier = model.named_steps["classifier"]
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_data)
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()

            if isinstance(shap_values, list):
                values = shap_values[1][0]
            elif getattr(shap_values, "ndim", 0) == 3:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]

            if hasattr(values, "values"):
                values = values.values

            plt.close("all")
            shap.bar_plot(values, feature_names=feature_names, max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            st.write("Positive values increase predicted attrition risk; negative values reduce it.")
        except Exception as exc:
            st.warning(f"SHAP explanation could not be rendered: {exc}")

st.markdown("---")
st.info("💡 End-to-end ML system: preprocessing, SMOTE, Random Forest, validation-based thresholding, FastAPI serving, Streamlit UI, and SHAP explainability.")
