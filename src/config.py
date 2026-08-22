import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hr_data.csv")
MODEL_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "model_pipeline.pkl")
SHAP_BACKGROUND_PATH = os.path.join(BASE_DIR, "models", "shap_background.pkl")
MODEL_METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

DEFAULT_MODEL_THRESHOLD = 0.30
CV_SPLITS = 5
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

# Threshold is selected on validation data, never on the final test set.
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01

RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [5, 10, None],
}


def load_threshold() -> float:
    """Load the threshold selected during training, with a safe fallback."""
    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        threshold = float(metadata.get("threshold", DEFAULT_MODEL_THRESHOLD))
        return min(max(threshold, THRESHOLD_MIN), THRESHOLD_MAX)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return DEFAULT_MODEL_THRESHOLD
