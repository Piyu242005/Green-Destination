import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features

st.set_page_config(page_title="Green Destinations | HR Intelligence", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1450px;padding-top:2rem}
.hero{padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(135deg,#0b1220,#164e63);color:white;margin-bottom:1.2rem}
.hero h1{margin:0}.hero p{opacity:.8;margin:.4rem 0 0}
.card{padding:1rem 1.1rem;border:1px solid rgba(128,128,128,.22);border-radius:16px;background:var(--background-color)}
.insight{padding:1rem;border:1px solid rgba(128,128,128,.18);border-radius:14px;background:rgba(128,128,128,.05);min-height:115px}
.footer{text-align:center;opacity:.65;margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(128,128,128,.18)}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🌱 Green Destinations</h1>
<p>HR Intelligence Center · Executive Overview · Risk Analytics · Explainable Decisions</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists(config.MODEL_PIPELINE_PATH): return None
    return joblib.load(config.MODEL_PIPELINE_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_PATH) if os.path.exists(config.DATA_PATH) else pd.DataFrame()

model, df = load_model(), load_data()
if model is None or df.empty:
    st.error("Model or HR dataset is unavailable. Run the training pipeline first.")
    st.stop()

threshold = float(config.load_threshold())

st.markdown("## 🏢 Executive Overview")
attrition_rate = (df.Attrition == "Yes").mean() * 100
high_risk = int((df.Attrition == "Yes").sum())
overtime_rate = (df.OverTime == "Yes").mean() * 100
avg_tenure = df.YearsAtCompany.mean()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Workforce", f"{len(df):,}")
c2.metric("Observed Attrition", f"{attrition_rate:.1f}%")
c3.metric("Overtime Exposure", f"{overtime_rate:.1f}%")
c4.metric("Average Tenure", f"{avg_tenure:.1f} yrs")

risk_tab, analytics_tab, employee_tab, whatif_tab, model_tab = st.tabs(["🔴 Risk Distribution", "📊 Workforce Analytics", "👤 Employee 360°", "🎛️ What-If Simulator", "⚙️ Model Health"])

with risk_tab:
    st.markdown("### Workforce Risk Distribution")
    observed = pd.DataFrame({"Status":["Retained","Attrition"],"Employees":[len(df)-high_risk,high_risk]}).set_index("Status")
    a,b = st.columns([1,1.5])
    with a:
        st.metric("Observed Attrition Cases", f"{high_risk:,}")
        st.metric("Observed Retention Cases", f"{len(df)-high_risk:,}")
    with b: st.bar_chart(observed, height=300)
    st.info("Observed attrition is not the same as predicted future risk. Use Employee 360° or What-If for model-based prediction.")

with analytics_tab:
    st.markdown("### Workforce Segmentation")
    f1,f2,f3 = st.columns(3)
    with f1: dept = st.selectbox("Department", ["All"] + sorted(df.Department.dropna().unique()))
    with f2: role = st.selectbox("Job Role", ["All"] + sorted(df.JobRole.dropna().unique()))
    with f3: overtime = st.selectbox("Overtime", ["All","Yes","No"])
    view=df.copy()
    if dept!="All": view=view[view.Department==dept]
    if role!="All": view=view[view.JobRole==role]
    if overtime!="All": view=view[view.OverTime==overtime]
    x1,x2,x3,x4=st.columns(4)
    x1.metric("Employees",f"{len(view):,}")
    x2.metric("Attrition",f"{(view.Attrition=='Yes').mean()*100:.1f}%" if len(view) else "0.0%")
    x3.metric("Avg Age",f"{view.Age.mean():.1f} yrs" if len(view) else "—")
    x4.metric("Avg Income",f"${view.MonthlyIncome.mean():,.0f}" if len(view) else "—")
    p1,p2=st.columns(2)
    with p1:
        st.markdown("**Department Attrition**")
        dchart=pd.crosstab(view.Department,view.Attrition,normalize='index').get('Yes',pd.Series(dtype=float))*100
        st.bar_chart(dchart,height=300)
    with p2:
        st.markdown("**Job Role Attrition**")
        rchart=pd.crosstab(view.JobRole,view.Attrition,normalize='index').get('Yes',pd.Series(dtype=float)).sort_values(ascending=False)*100
        st.bar_chart(rchart,height=300)
    st.markdown("### 💡 Retention Signals")
    i1,i2,i3=st.columns(3)
    top_dept=dchart.idxmax() if not dchart.empty else "N/A"
    top_role=rchart.idxmax() if not rchart.empty else "N/A"
    overtime_yes=view[view.OverTime=="Yes"]
    overtime_signal=(overtime_yes.Attrition=="Yes").mean()*100 if len(overtime_yes) else 0
    i1.markdown(f'<div class="insight"><b>Highest attrition department</b><h3>{top_dept}</h3><small>Observed rate in selected view.</small></div>',unsafe_allow_html=True)
    i2.markdown(f'<div class="insight"><b>Highest attrition role</b><h3>{top_role}</h3><small>Observed rate in selected view.</small></div>',unsafe_allow_html=True)
    i3.markdown(f'<div class="insight"><b>Overtime signal</b><h3>{overtime_signal:.1f}%</h3><small>Observed attrition among overtime employees.</small></div>',unsafe_allow_html=True)
    st.download_button("⬇️ Export filtered HR data",view.to_csv(index=False),"hr_analytics.csv","text/csv")

ROLES=["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"]

def predict(age, department, income, overtime, years, years_company, role, env_sat, job_sat):
    row={"Age":age,"Department":department,"MonthlyIncome":income,"OverTime":overtime,"TotalWorkingYears":years,"YearsAtCompany":years_company,"JobRole":role,"EnvironmentSatisfaction":env_sat,"JobSatisfaction":job_sat,"MonthlyRate":14313,"DailyRate":802,"HourlyRate":66,"BusinessTravel":"Travel_Rarely","DistanceFromHome":5,"Education":3,"EducationField":"Life Sciences","Gender":"Male","JobInvolvement":3,"JobLevel":2,"MaritalStatus":"Single","NumCompaniesWorked":1,"PercentSalaryHike":15,"PerformanceRating":3,"RelationshipSatisfaction":3,"StockOptionLevel":0,"TrainingTimesLastYear":3,"WorkLifeBalance":3,"YearsInCurrentRole":2,"YearsSinceLastPromotion":1,"YearsWithCurrManager":2}
    x=engineer_features(pd.DataFrame([row]))
    return float(model.predict_proba(x)[0][1])

with employee_tab:
    st.markdown("### 👤 Employee 360°")
    st.caption("Enter an employee profile to receive a model-based risk assessment and retention guidance.")
    e1,e2,e3=st.columns(3)
    with e1: e_age=st.slider("Age",18,60,30); e_dept=st.selectbox("Department",sorted(df.Department.dropna().unique()))
    with e2: e_income=st.number_input("Monthly Income (USD)",1000,20000,5000,500); e_role=st.selectbox("Job Role",ROLES)
    with e3: e_overtime=st.selectbox("Overtime",["Yes","No"]); e_tenure=st.slider("Years at Company",0,40,5)
    e4,e5=st.columns(2)
    with e4: e_env=st.slider("Environment Satisfaction",1,4,3)
    with e5: e_job=st.slider("Job Satisfaction",1,4,3)
    if st.button("Analyze Employee 360°",type="primary"):
        p=predict(e_age,e_dept,e_income,e_overtime,e_tenure,e_tenure,e_role,e_env,e_job)
        label="HIGH RISK" if p>=threshold else "LOW RISK"
        st.metric("Predicted Attrition Risk",f"{p*100:.1f}%",label)
        if p>=threshold:
            st.warning("Priority retention review recommended.")
            st.markdown("**Suggested actions:** review workload and overtime, conduct a 1:1 conversation, assess satisfaction, and review career progression.")
        else: st.success("Lower predicted risk. Continue normal engagement and periodic monitoring.")
        st.dataframe(pd.DataFrame({"Profile":["Age","Department","Role","Income","Overtime","Tenure"],"Value":[e_age,e_dept,e_role,f"${e_income:,.0f}",e_overtime,f"{e_tenure} yrs"]}),hide_index=True,use_container_width=True)

with whatif_tab:
    st.markdown("### 🎛️ What-If Attrition Simulator")
    st.caption("Change controllable factors and compare the model's predicted probability. This is a scenario tool, not a causal guarantee.")
    w1,w2=st.columns(2)
    with w1:
        st.markdown("**Current Profile**")
        wa_overtime=st.selectbox("Current overtime",["Yes","No"]); wa_job=st.slider("Current job satisfaction",1,4,2); wa_income=st.number_input("Current income (USD)",1000,20000,5000,500)
    with w2:
        st.markdown("**Scenario Profile**")
        wb_overtime=st.selectbox("Scenario overtime",["No","Yes"]); wb_job=st.slider("Scenario job satisfaction",1,4,4); wb_income=st.number_input("Scenario income (USD)",1000,20000,6500,500)
    if st.button("Compare Scenario",type="primary"):
        current=predict(30,"Sales",wa_income,wa_overtime,10,5,"Sales Executive",3,wa_job)
        scenario=predict(30,"Sales",wb_income,wb_overtime,10,5,"Sales Executive",3,wb_job)
        a,b,c=st.columns(3); a.metric("Current Risk",f"{current*100:.1f}%"); b.metric("Scenario Risk",f"{scenario*100:.1f}%"); c.metric("Change",f"{(scenario-current)*100:+.1f} pp")
        st.bar_chart(pd.DataFrame({"Risk %":[current*100,scenario*100]},index=["Current","Scenario"]))

with model_tab:
    st.markdown("### ⚙️ Model Health")
    st.caption("Operational checks for the deployed prediction artifact. Evaluation metrics should come from the latest training report.")
    m1,m2,m3=st.columns(3)
    m1.metric("Model Artifact","Available" if model else "Missing")
    m2.metric("Decision Threshold",f"{threshold:.2f}")
    m3.metric("Training Dataset",f"{len(df):,} rows")
    st.success("Model artifact loaded successfully. For production monitoring, connect this panel to logged prediction and drift metrics.")
    st.markdown("**Responsible AI:** predictions are decision-support only and should not independently determine hiring, termination, promotion or compensation decisions.")

st.markdown('<div class="footer">Built by <strong>Piyush Ramteke</strong> · Green Destinations HR Intelligence · Explainable ML</div>', unsafe_allow_html=True)
