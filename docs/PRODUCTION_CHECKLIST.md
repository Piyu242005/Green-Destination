# Production Checklist

- [x] Leakage-safe preprocessing and SMOTE pipeline
- [x] Validation-based threshold optimization
- [x] Final test-set evaluation
- [x] Model metadata generation
- [x] SHAP explainability
- [x] API schema validation (advanced API implementation)
- [x] Health endpoint (advanced API implementation)
- [x] Model comparison benchmark
- [x] Drift monitoring utility
- [x] Fairness/representation utility
- [x] HR analytics dashboard
- [x] Model card / responsible-AI guidance
- [ ] Run full training after checkout: `python src/train.py`
- [ ] Verify `models/model_metadata.json` from the current training run
- [ ] Run `pytest -q`
- [ ] Manually verify Streamlit and FastAPI deployments
- [ ] Establish a baseline prediction distribution for future drift checks
