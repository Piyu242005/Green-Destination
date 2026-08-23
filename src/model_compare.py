"""Compare baseline classifiers on the same leakage-safe feature pipeline."""
import os, sys
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features

def run_comparison():
    df = engineer_features(pd.read_csv(config.DATA_PATH))
    X = df.drop(["Attrition", "EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], axis=1)
    y = df["Attrition"].map({"Yes": 1, "No": 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=config.RANDOM_STATE)
    num = X.select_dtypes(include=["int64", "float64"]).columns
    cat = X.select_dtypes(include=["object"]).columns
    prep = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, random_state=42),
    }
    rows=[]
    for name, estimator in models.items():
        pipe = ImbPipeline([("preprocessor", prep), ("smote", SMOTE(random_state=42)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        p=pipe.predict_proba(X_test)[:,1]
        pred=(p>=.3).astype(int)
        rows.append({"model":name,"roc_auc":roc_auc_score(y_test,p),"pr_auc":average_precision_score(y_test,p),"recall":recall_score(y_test,p),"f1":f1_score(y_test,p)})
    result=pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    print(result.to_string(index=False))
    result.to_csv("model_comparison.csv", index=False)
    return result

if __name__ == "__main__": run_comparison()
