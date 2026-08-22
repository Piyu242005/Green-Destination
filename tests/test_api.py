from fastapi.testclient import TestClient
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import api_secure

client = TestClient(api_secure.app)

PAYLOAD = {
    "Age": 30, "Department": "Sales", "MonthlyIncome": 5000, "OverTime": "Yes",
    "TotalWorkingYears": 5, "YearsAtCompany": 2, "JobRole": "Sales Executive",
    "EnvironmentSatisfaction": 3, "JobSatisfaction": 3, "MonthlyRate": 14313,
    "DailyRate": 802, "HourlyRate": 66, "BusinessTravel": "Travel_Rarely",
    "DistanceFromHome": 5, "Education": 3, "EducationField": "Life Sciences",
    "Gender": "Male", "JobInvolvement": 3, "JobLevel": 2, "MaritalStatus": "Single",
    "NumCompaniesWorked": 1, "PercentSalaryHike": 15, "PerformanceRating": 3,
    "RelationshipSatisfaction": 3, "StockOptionLevel": 0, "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3, "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,
}


class FakeModel:
    def predict_proba(self, frame):
        assert "IncomePerAge" in frame.columns
        assert "TenureRatio" in frame.columns
        return [[0.2, 0.8]]


api_secure.model = FakeModel()


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["attrition_probability"] <= 1
    assert data["risk_level"] in {"High", "Low"}
    assert "tuned_threshold" in data


def test_predict_rejects_invalid_age():
    response = client.post("/predict", json={**PAYLOAD, "Age": 10})
    assert response.status_code == 422


def test_predict_rejects_missing_required_field():
    invalid = {key: value for key, value in PAYLOAD.items() if key != "MonthlyIncome"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
