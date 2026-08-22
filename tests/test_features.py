import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import engineer_features


def test_engineer_features_income_per_age():
    df = pd.DataFrame({"MonthlyIncome": [5000, 10000], "Age": [25, 50]})
    result = engineer_features(df)
    assert result["IncomePerAge"].tolist() == [200.0, 200.0]


def test_engineer_features_tenure_ratio():
    df = pd.DataFrame({"YearsAtCompany": [5, 10], "TotalWorkingYears": [9, 19]})
    result = engineer_features(df)
    assert result["TenureRatio"].tolist() == [0.5, 0.5]


def test_engineer_features_protects_against_zero_age():
    df = pd.DataFrame({"MonthlyIncome": [5000], "Age": [0]})
    result = engineer_features(df)
    assert result["IncomePerAge"].iloc[0] == 5000.0


def test_engineer_features_missing_columns():
    df = pd.DataFrame({"SomeColumn": [1, 2]})
    result = engineer_features(df)
    assert "IncomePerAge" not in result.columns
    assert "TenureRatio" not in result.columns
    assert len(result.columns) == 1
