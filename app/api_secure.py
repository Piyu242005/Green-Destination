from typing import Any
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features
from src.schemas import EmployeeInput

app = FastAPI(
    title="Green Destinations Employee Attrition API",
    description="Validated API for employee attrition probability and retention-risk scoring.",
    version="2.0.0",
)


def load_model():
    if not os.path.exists(config.MODEL_PIPELINE_PATH):
        return None
    try:
        return joblib.load(config.MODEL_PIPELINE_PATH)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return None


model = load_model()


def get_threshold() -> float:
    return config.load_threshold()


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "message": "Attrition Prediction API is active.",
        "developer": "Piyush Ramteke",
        "role": "Data Scientist | AI/ML Engineer",
        "status": "ready" if model is not None else "model_not_found",
        "threshold": get_threshold(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "healthy" if model is not None else "unhealthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: EmployeeInput) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run the training pipeline first.")

    try:
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_df = engineer_features(pd.DataFrame([payload]))
        probability = float(model.predict_proba(input_df)[0][1])
        threshold = get_threshold()
        risk_level = "High" if probability >= threshold else "Low"
        return {
            "attrition_probability": round(probability, 4),
            "risk_level": risk_level,
            "tuned_threshold": threshold,
            "action": "Immediate retention interview suggested" if risk_level == "High" else "Continue monitoring employee satisfaction",
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
