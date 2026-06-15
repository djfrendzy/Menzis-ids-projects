# Project 2 - Random Forest + Local Outlier Factor

## Summary
Detects DDoS network intrusions using a two-model pipeline:
- **Random Forest** (supervised) for high-precision DDoS classification
- **Local Outlier Factor - LOF** (unsupervised) for density-based anomaly detection
- **SHAP** for per-alert explainability

## Results (CIC-IDS-2017, n=33,862 test flows)

| Metric | Random Forest | LOF |
|---|---|---|
| Precision | 0.9998 | - |
| Recall | 0.9999 | - |
| F1 Score | 0.9999 | - |
| ROC-AUC | 1.0000 | 0.9187 |

> LOF is evaluated primarily by ROC-AUC as an unsupervised model - F1 is threshold-dependent.

## How to Run
```bash
pip install -r requirements.txt
python ids_rf_lof_pipeline.py
```
Update `CSV_PATH` in the script to point to your dataset file.

## Output Files
After running, the `outputs/` folder will contain:
| File | Description |
|---|---|
| `fig1_class_distribution.png` | Class balance chart |
| `fig2_feature_importance.png` | Correlation vs RF importance |
| `fig3_rf_confusion.png` | Random Forest confusion matrix |
| `fig4_lof_confusion.png` | LOF confusion matrix |
| `fig5_roc_curves.png` | ROC curves - both models |
| `fig6_shap_bar.png` | SHAP global importance |
| `fig7_shap_beeswarm.png` | SHAP direction of influence |
| `fig8_shap_ddos.png` | SHAP waterfall  DDoS example |
| `results_summary.json` | All numeric results |
