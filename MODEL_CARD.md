# Model Card — Green Destinations Attrition

## Intended use
Decision-support for HR analytics and employee-retention research. The model must not be used as an automatic employment, termination, promotion, or compensation decision-maker.

## Model
Random Forest with preprocessing, SMOTE, cross-validated hyperparameter search, validation-set F2 threshold selection, and SHAP explainability.

## Evaluation
Final evaluation is performed on an untouched stratified test set. Report ROC-AUC, PR-AUC, precision, recall, F1, F2, and confusion matrix in `models/model_metadata.json` after training.

## Risk and limitations
- Dataset is small and historical; performance may not generalize to another organization.
- Class imbalance means accuracy is not an appropriate primary metric.
- A probability is a model estimate, not a certainty.
- Demographic attributes require fairness review before operational use.
- Monitoring for data drift and prediction drift is required after deployment.

## Reproducibility
Run `python src/train.py` to regenerate the model and metadata with the configured random seed.
