"""Validated production-oriented API kept separate from the legacy demo API."""
import os, sys, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features

app=FastAPI(title="Green Destinations Attrition API",version="2.0.0")
try: model=joblib.load(config.MODEL_PIPELINE_PATH)
except Exception: model=None

class EmployeeInput(BaseModel):
    Age:int=Field(...,ge=18,le=80); Department:str; MonthlyIncome:float=Field(...,gt=0); OverTime:str
    TotalWorkingYears:int=Field(...,ge=0); YearsAtCompany:int=Field(...,ge=0); JobRole:str
    EnvironmentSatisfaction:int=Field(...,ge=1,le=4); JobSatisfaction:int=Field(...,ge=1,le=4)
    MonthlyRate:float=Field(...,gt=0); DailyRate:float=Field(...,gt=0); HourlyRate:float=Field(...,gt=0)
    BusinessTravel:str; DistanceFromHome:float=Field(...,ge=0); Education:int=Field(...,ge=1,le=5); EducationField:str; Gender:str
    JobInvolvement:int=Field(...,ge=1,le=4); JobLevel:int=Field(...,ge=1,le=5); MaritalStatus:str
    NumCompaniesWorked:int=Field(...,ge=0); PercentSalaryHike:float=Field(...,ge=0,le=100); PerformanceRating:int=Field(...,ge=1,le=5)
    RelationshipSatisfaction:int=Field(...,ge=1,le=4); StockOptionLevel:int=Field(...,ge=0,le=3); TrainingTimesLastYear:int=Field(...,ge=0)
    WorkLifeBalance:int=Field(...,ge=1,le=4); YearsInCurrentRole:int=Field(...,ge=0); YearsSinceLastPromotion:int=Field(...,ge=0); YearsWithCurrManager:int=Field(...,ge=0)

@app.get("/health")
def health(): return {"status":"healthy" if model else "degraded","model_loaded":model is not None}

@app.post("/predict")
def predict(data:EmployeeInput):
    if model is None: raise HTTPException(503,"Model not loaded")
    try:
        x=engineer_features(pd.DataFrame([data.model_dump()]))
        p=float(model.predict_proba(x)[0][1]); threshold=config.load_threshold()
        risk="High" if p>=threshold else "Low"
        return {"attrition_probability":round(p,4),"risk_level":risk,"threshold":threshold,"action":"Targeted retention review" if risk=="High" else "Continue engagement monitoring"}
    except Exception as exc: raise HTTPException(422,f"Prediction failed: {exc}") from exc
