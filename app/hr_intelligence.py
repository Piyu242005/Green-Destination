import os
import sys
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features

st.set_page_config(page_title="Green Destinations | HR Intelligence", page_icon="🌱", layout="wide")
st.markdown("""<style>.block-container{max-width:1450px;padding-top:1.8rem}.hero{padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(135deg,#0b1220,#164e63);color:white;margin-bottom:1.2rem}.hero h1{margin:0}.hero p{opacity:.8;margin:.4rem 0 0}.insight{padding:1rem;border:1px solid rgba(128,128,128,.18);border-radius:14px;background:rgba(128,128,128,.05);min-height:110px}.footer{text-align:center;opacity:.65;margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(128,128,128,.18)}</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌱 Green Destinations</h1><p>HR Intelligence Center · Executive Overview · Risk Analytics · Explainable Decisions</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PIPELINE_PATH) if os.path.exists(config.MODEL_PIPELINE_PATH) else None
@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_PATH) if os.path.exists(config.DATA_PATH) else pd.DataFrame()
model, df = load_model(), load_data()
if model is None or df.empty: st.error("Model or HR dataset is unavailable. Run the training pipeline first."); st.stop()
threshold=float(config.load_threshold())

def plot(fig,height=350):
    fig.update_layout(height=height,margin=dict(l=20,r=20,t=55,b=20),template="plotly_white",legend=dict(orientation="h",y=1.08))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"responsive":True})

attrition=(df.Attrition=="Yes").mean()*100; attrition_cases=int((df.Attrition=="Yes").sum())
st.markdown("## 🏢 Executive Overview")
a,b,c,d=st.columns(4); a.metric("Workforce",f"{len(df):,}"); b.metric("Observed Attrition",f"{attrition:.1f}%"); c.metric("Overtime Exposure",f"{(df.OverTime=='Yes').mean()*100:.1f}%"); d.metric("Average Tenure",f"{df.YearsAtCompany.mean():.1f} yrs")

tabs=st.tabs(["🔴 Risk Distribution","📊 Workforce Analytics","🔗 Relationships","🕸️ Engagement","👤 Employee 360°","🎛️ What-If","⚙️ Model Health"])

with tabs[0]:
    st.markdown("### Risk Distribution")
    risk=pd.DataFrame({"Status":["Retained","Attrition"],"Employees":[len(df)-attrition_cases,attrition_cases]})
    x,y=st.columns([1,1.5]); x.metric("Attrition Cases",f"{attrition_cases:,}"); x.metric("Retention Cases",f"{len(df)-attrition_cases:,}")
    with y: plot(px.pie(risk,names="Status",values="Employees",hole=.62,title="Retention vs Attrition"),360)
    dep=pd.crosstab(df.Department,df.Attrition,normalize="index").get("Yes",pd.Series(dtype=float)).reset_index(name="Rate"); dep.Rate*=100
    plot(px.bar(dep,x="Department",y="Rate",text_auto=".1f",title="Observed Attrition by Department"))

with tabs[1]:
    st.markdown("### Workforce Segmentation")
    f1,f2,f3=st.columns(3)
    with f1: sd=st.selectbox("Department",["All"]+sorted(df.Department.dropna().unique()))
    with f2: sr=st.selectbox("Job Role",["All"]+sorted(df.JobRole.dropna().unique()))
    with f3: so=st.selectbox("Overtime",["All","Yes","No"])
    v=df.copy()
    if sd!="All": v=v[v.Department==sd]
    if sr!="All": v=v[v.JobRole==sr]
    if so!="All": v=v[v.OverTime==so]
    q1,q2,q3,q4=st.columns(4); q1.metric("Employees",f"{len(v):,}"); q2.metric("Attrition",f"{(v.Attrition=='Yes').mean()*100:.1f}%" if len(v) else "0%"); q3.metric("Avg Age",f"{v.Age.mean():.1f} yrs" if len(v) else "—"); q4.metric("Avg Income",f"${v.MonthlyIncome.mean():,.0f}" if len(v) else "—")
    dep=pd.crosstab(v.Department,v.Attrition,normalize="index").get("Yes",pd.Series(dtype=float)).sort_values(ascending=False).reset_index(name="Rate"); dep.Rate*=100
    role=pd.crosstab(v.JobRole,v.Attrition,normalize="index").get("Yes",pd.Series(dtype=float)).sort_values(ascending=False).reset_index(name="Rate"); role.Rate*=100
    x,y=st.columns(2)
    with x: plot(px.bar(dep,x="Rate",y="Department",orientation="h",text_auto=".1f",title="Attrition by Department"),330)
    with y: plot(px.bar(role,x="Rate",y="JobRole",orientation="h",text_auto=".1f",title="Attrition by Job Role"),330)
    x,y=st.columns(2)
    with x:
        ot=pd.crosstab(v.OverTime,v.Attrition,normalize="index").get("Yes",pd.Series(dtype=float)).reset_index(name="Rate"); ot.Rate*=100
        plot(px.bar(ot,x="OverTime",y="Rate",text_auto=".1f",title="Attrition by Overtime"),310)
    with y:
        av=v.copy(); av["Age Group"]=pd.cut(av.Age,[17,25,35,45,60,100],labels=["18–25","26–35","36–45","46–60","60+"])
        ag=pd.crosstab(av["Age Group"],av.Attrition,normalize="index").get("Yes",pd.Series(dtype=float)).reset_index(name="Rate"); ag.Rate*=100
        plot(px.line(ag,x="Age Group",y="Rate",markers=True,title="Attrition by Age Group"),310)
    i1,i2,i3=st.columns(3)
    i1.markdown(f'<div class="insight"><b>Highest attrition department</b><h3>{dep.iloc[0].Department if len(dep) else "N/A"}</h3></div>',unsafe_allow_html=True)
    i2.markdown(f'<div class="insight"><b>Highest attrition role</b><h3>{role.iloc[0].JobRole if len(role) else "N/A"}</h3></div>',unsafe_allow_html=True)
    i3.markdown(f'<div class="insight"><b>Overtime attrition</b><h3>{(v[v.OverTime=="Yes"].Attrition=="Yes").mean()*100:.1f}%</h3></div>',unsafe_allow_html=True)
    st.download_button("⬇️ Export filtered HR data",v.to_csv(index=False),"hr_analytics.csv","text/csv")

with tabs[2]:
    st.markdown("### Relationship & Distribution Analysis")
    x,y=st.columns(2)
    with x: plot(px.scatter(df,x="Age",y="MonthlyIncome",color="Attrition",hover_data=["Department","JobRole","YearsAtCompany"],title="Age vs Monthly Income"),360)
    with y: plot(px.scatter(df,x="YearsAtCompany",y="MonthlyIncome",color="Attrition",hover_data=["Age","Department","JobRole"],title="Tenure vs Monthly Income"),360)
    x,y=st.columns(2)
    with x: plot(px.box(df,x="Attrition",y="MonthlyIncome",color="Attrition",points="outliers",title="Income Distribution by Attrition"),350)
    with y: plot(px.violin(df,x="Attrition",y="YearsAtCompany",color="Attrition",box=True,points=False,title="Tenure Distribution by Attrition"),350)
    nums=[c for c in ["Age","MonthlyIncome","TotalWorkingYears","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","JobSatisfaction","EnvironmentSatisfaction","WorkLifeBalance","JobInvolvement","DistanceFromHome"] if c in df.columns]
    plot(px.imshow(df[nums].corr(),text_auto=".2f",aspect="auto",title="HR Factor Correlation Heatmap"),520)
    x,y=st.columns(2)
    with x: plot(px.histogram(df,x="Age",color="Attrition",nbins=18,barmode="overlay",title="Age Distribution"),330)
    with y: plot(px.histogram(df,x="MonthlyIncome",color="Attrition",nbins=25,barmode="overlay",title="Monthly Income Distribution"),330)

with tabs[3]:
    st.markdown("### 🕸️ Engagement & Satisfaction")
    mapping={"Job Satisfaction":"JobSatisfaction","Environment Satisfaction":"EnvironmentSatisfaction","Work-Life Balance":"WorkLifeBalance","Job Involvement":"JobInvolvement","Relationship Satisfaction":"RelationshipSatisfaction"}; mapping={k:v for k,v in mapping.items() if v in df.columns}
    labels=list(mapping); vals=[df[c].mean() for c in mapping.values()]
    fig=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=labels+[labels[0]],fill="toself",name="Workforce Average")); fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,4])),title="Workforce Engagement Radar",height=500); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    long=df[list(mapping.values())].mean().reset_index(); long.columns=["Factor","Score"]; long["Factor"]=long.Factor.map({v:k for k,v in mapping.items()}); plot(px.bar(long,x="Score",y="Factor",orientation="h",text_auto=".2f",title="Average Engagement Scores"),340)
    sat=df.groupby("Attrition")[list(mapping.values())].mean().T.reset_index(); sat.columns=["Factor","Retained","Attrition"]; sat["Factor"]=sat.Factor.map({v:k for k,v in mapping.items()}); plot(px.bar(sat,x="Factor",y=["Retained","Attrition"],barmode="group",title="Engagement by Attrition Outcome"),340)

ROLES=sorted(df.JobRole.dropna().unique())
def predict(age,department,income,overtime,years,role,env,job):
    row={"Age":age,"Department":department,"MonthlyIncome":income,"OverTime":overtime,"TotalWorkingYears":years,"YearsAtCompany":years,"JobRole":role,"EnvironmentSatisfaction":env,"JobSatisfaction":job,"MonthlyRate":14313,"DailyRate":802,"HourlyRate":66,"BusinessTravel":"Travel_Rarely","DistanceFromHome":5,"Education":3,"EducationField":"Life Sciences","Gender":"Male","JobInvolvement":3,"JobLevel":2,"MaritalStatus":"Single","NumCompaniesWorked":1,"PercentSalaryHike":15,"PerformanceRating":3,"RelationshipSatisfaction":3,"StockOptionLevel":0,"TrainingTimesLastYear":3,"WorkLifeBalance":3,"YearsInCurrentRole":2,"YearsSinceLastPromotion":1,"YearsWithCurrManager":2}; return float(model.predict_proba(engineer_features(pd.DataFrame([row])))[0][1])

with tabs[4]:
    st.markdown("### 👤 Employee 360°")
    a,b,c=st.columns(3)
    with a: age=st.slider("Age",18,60,30); department=st.selectbox("Department",sorted(df.Department.dropna().unique()))
    with b: income=st.number_input("Monthly Income (USD)",1000,20000,5000,500); role=st.selectbox("Job Role",ROLES)
    with c: overtime=st.selectbox("Overtime",["Yes","No"]); years=st.slider("Years at Company",0,40,5)
    a,b=st.columns(2)
    with a: env=st.slider("Environment Satisfaction",1,4,3)
    with b: job=st.slider("Job Satisfaction",1,4,3)
    if st.button("Analyze Employee 360°",type="primary"):
        p=predict(age,department,income,overtime,years,role,env,job); st.metric("Predicted Attrition Risk",f"{p*100:.1f}%","HIGH RISK" if p>=threshold else "LOW RISK"); (st.warning if p>=threshold else st.success)("Priority retention review recommended." if p>=threshold else "Lower predicted risk. Continue normal engagement and monitoring.")

with tabs[5]:
    st.markdown("### 🎛️ What-If Attrition Simulator")
    a,b=st.columns(2)
    with a: co=st.selectbox("Current overtime",["Yes","No"]); cj=st.slider("Current job satisfaction",1,4,2); ci=st.number_input("Current income (USD)",1000,20000,5000,500)
    with b: so=st.selectbox("Scenario overtime",["No","Yes"]); sj=st.slider("Scenario job satisfaction",1,4,4); si=st.number_input("Scenario income (USD)",1000,20000,6500,500)
    if st.button("Compare Scenario",type="primary"):
        p1=predict(30,"Sales",ci,co,10,"Sales Executive",3,cj); p2=predict(30,"Sales",si,so,10,"Sales Executive",3,sj); a,b,c=st.columns(3); a.metric("Current Risk",f"{p1*100:.1f}%"); b.metric("Scenario Risk",f"{p2*100:.1f}%"); c.metric("Change",f"{(p2-p1)*100:+.1f} pp"); plot(px.bar(pd.DataFrame({"Profile":["Current","Scenario"],"Risk":[p1*100,p2*100]}),x="Profile",y="Risk",text_auto=".1f",title="Scenario Risk Comparison"),330)

with tabs[6]:
    st.markdown("### ⚙️ Model Health")
    a,b,c=st.columns(3); a.metric("Model Artifact","Available"); b.metric("Decision Threshold",f"{threshold:.2f}"); c.metric("Training Dataset",f"{len(df):,} rows"); st.success("Model artifact loaded successfully. Connect logged prediction and drift metrics for production monitoring."); st.info("Responsible AI: predictions are decision-support only and should not independently determine employment decisions.")

st.markdown('<div class="footer">Built by <strong>Piyush Ramteke</strong> · Green Destinations HR Intelligence · Explainable ML</div>',unsafe_allow_html=True)
