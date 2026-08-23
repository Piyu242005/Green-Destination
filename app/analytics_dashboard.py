"""HR analytics and monitoring dashboard."""
import os, sys
import pandas as pd
import streamlit as st
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.monitoring import group_metrics

st.set_page_config(page_title="Green Destinations HR Analytics", layout="wide")
st.title("🌱 Green Destinations — HR Analytics & Monitoring")
df=pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","hr_data.csv"))

attrition_rate=(df["Attrition"]=="Yes").mean()*100
c1,c2,c3,c4=st.columns(4)
c1.metric("Employees",len(df)); c2.metric("Attrition Rate",f"{attrition_rate:.1f}%")
c3.metric("High Overtime",f"{(df['OverTime']=='Yes').mean()*100:.1f}%")
c4.metric("Avg Income",f"${df['MonthlyIncome'].mean():,.0f}")

st.subheader("Attrition by Department")
st.bar_chart(pd.crosstab(df["Department"],df["Attrition"],normalize="index")["Yes"]*100)
st.subheader("Attrition by Job Role")
st.bar_chart(pd.crosstab(df["JobRole"],df["Attrition"],normalize="index")["Yes"]*100)
st.subheader("Key Drivers")
st.write("Use this dashboard for descriptive HR analytics. Predictions should support, not automatically determine, employment decisions.")
if "Gender" in df.columns:
    # Descriptive fairness view; prediction column is intentionally required before predictive parity metrics are shown.
    st.subheader("Fairness / Representation Check")
    st.dataframe(df.groupby("Gender")["Attrition"].apply(lambda s:(s=="Yes").mean()).rename("actual_attrition_rate").reset_index())
