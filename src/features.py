import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature engineering shared by training and inference."""
    result = df.copy()

    if "MonthlyIncome" in result.columns and "Age" in result.columns:
        age = pd.to_numeric(result["Age"], errors="coerce").clip(lower=1)
        income = pd.to_numeric(result["MonthlyIncome"], errors="coerce")
        result["IncomePerAge"] = income / age

    if "YearsAtCompany" in result.columns and "TotalWorkingYears" in result.columns:
        tenure = pd.to_numeric(result["YearsAtCompany"], errors="coerce").clip(lower=0)
        total = pd.to_numeric(result["TotalWorkingYears"], errors="coerce").clip(lower=0)
        result["TenureRatio"] = tenure / (total + 1)

    return result
