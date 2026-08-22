import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.features import engineer_features


def build_advanced_pipeline(X):
    """Build preprocessing, SMOTE and Random Forest in one leakage-safe pipeline."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight=None)),
    ])


def select_threshold(y_true, probabilities):
    """Select a threshold on validation data using F2, favoring recall."""
    best_threshold = config.DEFAULT_MODEL_THRESHOLD
    best_score = -1.0

    thresholds = np.arange(
        config.THRESHOLD_MIN,
        config.THRESHOLD_MAX + config.THRESHOLD_STEP / 2,
        config.THRESHOLD_STEP,
    )
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, predictions, beta=2, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 2))

    return best_threshold, best_score


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, predictions, beta=2, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def train_and_save():
    print("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    df = engineer_features(df)

    X = df.drop(["Attrition", "EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], axis=1)
    y = df["Attrition"].map({"Yes": 1, "No": 0})

    # Keep a final untouched test set for unbiased evaluation.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    validation_fraction = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        stratify=y_train_val,
        random_state=config.RANDOM_STATE,
    )

    print("Training model with SMOTE and Random Forest...")
    pipeline = build_advanced_pipeline(X)
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=config.RF_PARAM_GRID,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Threshold selection happens only on validation data.
    val_probabilities = best_model.predict_proba(X_val)[:, 1]
    threshold, validation_f2 = select_threshold(y_val, val_probabilities)

    # Final metrics are calculated once on the untouched test set.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, threshold)
    print("\n--- Final Test Performance ---")
    print(classification_report(y_test, (test_probabilities >= threshold).astype(int), zero_division=0))
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Selected threshold: {threshold:.2f}")

    os.makedirs(os.path.dirname(config.MODEL_PIPELINE_PATH), exist_ok=True)
    joblib.dump(best_model, config.MODEL_PIPELINE_PATH)
    X_train.head(100).to_pickle(config.SHAP_BACKGROUND_PATH)

    metadata = {
        "model": "RandomForestClassifier",
        "best_params": grid_search.best_params_,
        "threshold": threshold,
        "validation_f2": round(float(validation_f2), 4),
        "test_metrics": test_metrics,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "random_state": config.RANDOM_STATE,
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Model saved to {config.MODEL_PIPELINE_PATH}")
    print(f"Metadata saved to {config.MODEL_METADATA_PATH}")


if __name__ == "__main__":
    train_and_save()
