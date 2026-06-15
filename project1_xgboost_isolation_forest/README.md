# Project 1 — XGBoost + Isolation Forest

## Summary
Detects DDoS network intrusions using a two-model pipeline:
- **XGBoost** (supervised) for known attack classification
- **Isolation Forest** (unsupervised) for zero-day anomaly detection
- **SHAP** for per-alert explainability

## Results (CIC-IDS-2017, n=33,862 test flows)

| Metric | XGBoost | Isolation Forest |
|---|---|---|
| Precision | 0.9999 | 0.9387 |
| Recall | 0.9999 | 0.6362 |
| F1 Score | 0.9999 | 0.7584 |
| ROC-AUC | 1.0000 | 0.8265 |

## How to Run
```bash
pip install -r requirements.txt
python ids_pipeline.py
```
Update `CSV_PATH` in the script to point to your dataset file.

## Output Files
After running, the `outputs/` folder will contain:
| File | Description |
|---|---|
| `fig1_class_distribution.png` | Class balance chart |
| `fig2_feature_importance.png` | Correlation vs RF importance |
| `fig3_xgb_confusion.png` | XGBoost confusion matrix |
| `fig4_iso_confusion.png` | Isolation Forest confusion matrix |
| `fig5_roc_curves.png` | ROC curves — both models |
| `fig6_shap_bar.png` | SHAP global importance |
| `fig7_shap_beeswarm.png` | SHAP direction of influence |
| `fig8_shap_ddos.png` | SHAP waterfall — DDoS example |
| `results_summary.json` | All numeric results |
