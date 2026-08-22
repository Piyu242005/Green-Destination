from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    """Validated employee attributes accepted by the prediction API."""

    Age: int = Field(..., ge=18, le=100)
    Department: str
    MonthlyIncome: float = Field(..., gt=0)
    OverTime: str
    TotalWorkingYears: float = Field(..., ge=0)
    YearsAtCompany: float = Field(..., ge=0)
    JobRole: str
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MonthlyRate: float = Field(..., ge=0)
    DailyRate: float = Field(..., ge=0)
    HourlyRate: float = Field(..., ge=0)
    BusinessTravel: str
    DistanceFromHome: float = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EducationField: str
    Gender: str
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    MaritalStatus: str
    NumCompaniesWorked: int = Field(..., ge=0)
    PercentSalaryHike: float = Field(..., ge=0, le=100)
    PerformanceRating: int = Field(..., ge=1, le=5)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TrainingTimesLastYear: int = Field(..., ge=0)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsInCurrentRole: float = Field(..., ge=0)
    YearsSinceLastPromotion: float = Field(..., ge=0)
    YearsWithCurrManager: float = Field(..., ge=0)
