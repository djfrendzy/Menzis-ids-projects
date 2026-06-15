import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # no display needed, saves files directly
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json
warnings.filterwarnings('ignore')

# OUTPUT FOLDER 
# All figures and results will be saved here
OUTPUT = "outputs"
os.makedirs(OUTPUT, exist_ok=True)

#  FILE PATH 
CSV_PATH = "Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv"

# SECTION 1: Load & Explore the Real Dataset
print("SECTION 1- Data Exploration & Preparation")
print("-" * 40)

# 1.1 Load CSV
# CICFlowMeter adds leading spaces to column names - strip them all
df = pd.read_csv(CSV_PATH, encoding='utf-8', low_memory=False)
df.columns = df.columns.str.strip()

print(f"\nRaw dataset shape : {df.shape}")
print(f"\nClass distribution:")
print(df['Label'].value_counts())
print(f"\nClass balance (%):")
print(df['Label'].value_counts(normalize=True).mul(100).round(2))

# 1.2 Handle infinities (caused by zero-duration flows - division by zero)
# Replace inf/-inf with NaN first, then fill with column median
df.replace([np.inf, -np.inf], np.nan, inplace=True)
missing_before = df.isnull().sum()
print(f"\nColumns with missing values before cleaning:")
print(missing_before[missing_before > 0])

# Fill missing values with the column median (robust to outliers)
df.fillna(df.median(numeric_only=True), inplace=True)
print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")

# 1.3 Class distribution bar chart
fig, ax = plt.subplots(figsize=(7, 4))
counts = df['Label'].value_counts()
colors = ['#2196F3', '#F44336']
counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.5)
ax.set_title('Class Distribution - CIC-IDS-2017 DDoS Dataset', fontsize=13, fontweight='bold')
ax.set_xlabel('Traffic Class')
ax.set_ylabel('Number of Flows')
ax.tick_params(axis='x', rotation=0)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig1_class_distribution.png", dpi=150)
plt.close()
print(f"\nSaved - fig1_class_distribution.png")

# 1.4 Binary label: BENIGN = 0, DDoS = 1
df['binary_label'] = (df['Label'] == 'DDoS').astype(int)

# 1.5 Identify all numeric feature columns (exclude the label columns)
FEATURE_COLS = [c for c in df.columns if c not in ['Label', 'binary_label']]
print(f"\nTotal numeric features available: {len(FEATURE_COLS)}")

# 1.6 Stratified Train / Validation / Test split - 70 / 15 / 15
# Stratified = each split has the same BENIGN/DDoS ratio
# Scaler fit ONLY on train → prevents data leakage into val/test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[FEATURE_COLS]
y = df['binary_label']

# First cut: 85% temp, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)

# Second cut: split the 85% into 70% train + 15% val
# 0.1765 × 0.85 ≈ 0.15 of total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

print(f"\nSplit sizes:")
print(f"  Train : {len(X_train):,} flows")
print(f"  Val   : {len(X_val):,}  flows")
print(f"  Test  : {len(X_test):,} flows")

# Scale features - fit ONLY on training data
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),  columns=FEATURE_COLS)
X_val_sc   = pd.DataFrame(scaler.transform(X_val),        columns=FEATURE_COLS)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),       columns=FEATURE_COLS)

# Reset all indices so boolean indexing works correctly later
y_train = y_train.reset_index(drop=True)
y_val   = y_val.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# SECTION 2 - Feature Engineering & Selection
print("SECTION 2 - Feature Engineering & Selection")
print("-" * 40)

#  Technique 1: Pearson Correlation with binary label 
corr = X_train_sc.corrwith(y_train).abs().sort_values(ascending=False)
print(f"\nTop 15 features by |Pearson correlation| with DDoS label:")
print(corr.head(15).to_string())

# Technique 2: Random Forest Feature Importance (Gini) 
from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest for feature importance (this may take some mins)...")
rf_selector = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,          # use all CPU cores
    random_state=42,
    max_depth=10        # limit depth to speed up selection step
)
rf_selector.fit(X_train_sc, y_train)
rf_imp = pd.Series(rf_selector.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\nTop 15 features by Random Forest Gini importance:")
print(rf_imp.head(15).to_string())

# Plot both side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

corr.head(15).sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Feature Importance - Pearson Correlation\n(with DDoS label)', fontweight='bold')
axes[0].set_xlabel('|Pearson r|')

rf_imp.head(15).sort_values().plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title('Feature Importance - Random Forest\n(Gini Impurity)', fontweight='bold')
axes[1].set_xlabel('Mean Decrease in Impurity')

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig2_feature_importance.png", dpi=150)
plt.close()
print("\nSaved → fig2_feature_importance.png")

# Select the union of top 15 features from both methods
top_corr = set(corr.head(15).index)
top_rf   = set(rf_imp.head(15).index)
SELECTED = sorted(list(top_corr | top_rf))
print(f"\nFinal selected feature set ({len(SELECTED)} features):")
for f in SELECTED:
    print(f"  • {f}")

# Apply selection
X_train_sel = X_train_sc[SELECTED].reset_index(drop=True)
X_val_sel   = X_val_sc[SELECTED].reset_index(drop=True)
X_test_sel  = X_test_sc[SELECTED].reset_index(drop=True)

# SECTION 3 - Model Development
print("SECTION 3 - Model Development")
print("-" * 40)

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support,
    ConfusionMatrixDisplay, roc_curve
)
import xgboost as xgb

# MODEL A: XGBoost (Supervised) 
print("\n Model A: XGBoost (Supervised)")

# scale_pos_weight handles class imbalance:
# = number of BENIGN samples / number of DDoS samples in training set
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pw = neg / pos
print(f"  Class ratio (scale_pos_weight): {scale_pw:.3f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,       # number of boosting trees
    max_depth=6,            # maximum tree depth
    learning_rate=0.1,      # step size shrinkage
    subsample=0.8,          # fraction of rows per tree
    colsample_bytree=0.8,   # fraction of features per tree
    scale_pos_weight=scale_pw,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_sel, y_train,
    eval_set=[(X_val_sel, y_val)],
    verbose=False
)

y_pred_xgb = xgb_model.predict(X_test_sel)
y_prob_xgb = xgb_model.predict_proba(X_test_sel)[:, 1]

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_xgb, average='binary')
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

print(f"\n  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc_xgb:.4f}")
print(f"\n{classification_report(y_test, y_pred_xgb, target_names=['BENIGN','DDoS'])}")

# XGBoost Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_xgb,
    display_labels=['BENIGN', 'DDoS'],
    cmap='Blues', ax=ax
)
ax.set_title('XGBoost – Confusion Matrix (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig3_xgb_confusion.png", dpi=150)
plt.close()
print("Saved → fig3_xgb_confusion.png")

# MODEL B: Isolation Forest (Unsupervised) 
print("\n Model B: Isolation Forest (Unsupervised) ")

from sklearn.ensemble import IsolationForest

# Train ONLY on benign flows, the model learns what "normal" looks like
X_train_benign = X_train_sel[y_train == 0].reset_index(drop=True)
print(f"  Training on {len(X_train_benign):,} BENIGN flows only.")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # expect ~5% anomalies in live traffic
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_benign)

# Predict: Isolation Forest returns +1 (normal) or -1 (anomaly)
iso_raw    = iso_forest.predict(X_test_sel)
y_pred_iso = np.where(iso_raw == -1, 1, 0)   # convert to 0/1

prec_i, rec_i, f1_i, _ = precision_recall_fscore_support(y_test, y_pred_iso, average='binary')
scores_iso = -iso_forest.decision_function(X_test_sel)  # flip: higher = more anomalous
auc_iso    = roc_auc_score(y_test, scores_iso)

print(f"\n  Precision : {prec_i:.4f}")
print(f"  Recall    : {rec_i:.4f}")
print(f"  F1 Score  : {f1_i:.4f}")
print(f"  ROC-AUC   : {auc_iso:.4f}")
print(f"\n{classification_report(y_test, y_pred_iso, target_names=['BENIGN','DDoS'])}")

# Isolation Forest Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_iso,
    display_labels=['BENIGN', 'DDoS'],
    cmap='Oranges', ax=ax
)
ax.set_title('Isolation Forest – Confusion Matrix (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig4_iso_confusion.png", dpi=150)
plt.close()
print("Saved → fig4_iso_confusion.png")

#  ROC Curves: both models on one chart 
fpr_x, tpr_x, _ = roc_curve(y_test, y_prob_xgb)
fpr_i, tpr_i, _ = roc_curve(y_test, scores_iso)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_x, tpr_x, color='steelblue',  lw=2, label=f'XGBoost        (AUC = {auc_xgb:.4f})')
ax.plot(fpr_i, tpr_i, color='darkorange', lw=2, label=f'Isolation Forest (AUC = {auc_iso:.4f})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curves – XGBoost vs Isolation Forest', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig5_roc_curves.png", dpi=150)
plt.close()
print("Saved → fig5_roc_curves.png")

#  Model Comparison Table 
comparison = pd.DataFrame({
    'Model':     ['XGBoost (Supervised)', 'Isolation Forest (Unsupervised)'],
    'Precision': [round(prec,  4), round(prec_i, 4)],
    'Recall':    [round(rec,   4), round(rec_i,  4)],
    'F1':        [round(f1,    4), round(f1_i,   4)],
    'ROC-AUC':   [round(auc_xgb, 4), round(auc_iso, 4)],
})
print(f"\nModel Comparison:\n{comparison.to_string(index=False)}")


# SECTION 4  SHAP Explainability
print("SECTION 4 - SHAP Explainability")
print("-" * 40)

import shap

# Use a random sample of 1,000 test flows for speed
X_shap  = X_test_sel.sample(1000, random_state=42).reset_index(drop=True)
y_shap  = y_test.iloc[X_test_sel.sample(1000, random_state=42).index].reset_index(drop=True)

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap)

# 4.1 Global bar chart; which features drive DDoS predictions most?
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type='bar',
                  show=False, max_display=15, color='#1976D2')
plt.title('SHAP Feature Importance (Global - DDoS Detection)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig6_shap_summary_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved → fig6_shap_summary_bar.png")

# 4.2 Beeswarm; direction of feature influence
plt.figure()
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.title('SHAP Beeswarm: Feature Impact Direction', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig7_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved → fig7_shap_beeswarm.png")

# 4.3 Individual explanation — Example 1: Confirmed DDoS alert
tp_mask = (y_shap == 1) & (xgb_model.predict(X_shap) == 1)
if tp_mask.any():
    ex1 = tp_mask.idxmax()   # first true positive
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[ex1],
            base_values=explainer.expected_value,
            data=X_shap.iloc[ex1],
            feature_names=list(X_shap.columns)
        ), show=False, max_display=12
    )
    plt.title('SHAP Waterfall - Example 1: Confirmed DDoS Alert', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/fig8_shap_ddos_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved → fig8_shap_ddos_example.png")

# 4.4 Individual explanation: Example 2: False Positive
fp_mask = (y_shap == 0) & (xgb_model.predict(X_shap) == 1)
if fp_mask.any():
    ex2 = fp_mask.idxmax()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[ex2],
            base_values=explainer.expected_value,
            data=X_shap.iloc[ex2],
            feature_names=list(X_shap.columns)
        ), show=False, max_display=12
    )
    plt.title('SHAP Waterfall - Example 2: False Positive Investigation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/fig9_shap_fp_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved → fig9_shap_fp_example.png")
else:
    print("No false positives found in SHAP sample; XGBoost is extremely precise.")


# SECTION 5 - Save Results Summary
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_iso = confusion_matrix(y_test, y_pred_iso)

results = {
    "dataset": {
        "total_flows": int(len(df)),
        "benign": int((df['Label'] == 'BENIGN').sum()),
        "ddos":   int((df['Label'] == 'DDoS').sum()),
        "features_total": len(FEATURE_COLS),
        "features_selected": len(SELECTED),
        "selected_feature_names": SELECTED
    },
    "splits": {
        "train": int(len(X_train)),
        "val":   int(len(X_val)),
        "test":  int(len(X_test))
    },
    "xgboost": {
        "precision": round(prec,    4),
        "recall":    round(rec,     4),
        "f1":        round(f1,      4),
        "roc_auc":   round(auc_xgb, 4),
        "TP": int(cm_xgb[1,1]), "FP": int(cm_xgb[0,1]),
        "TN": int(cm_xgb[0,0]), "FN": int(cm_xgb[1,0])
    },
    "isolation_forest": {
        "precision": round(prec_i,  4),
        "recall":    round(rec_i,   4),
        "f1":        round(f1_i,    4),
        "roc_auc":   round(auc_iso, 4),
        "TP": int(cm_iso[1,1]), "FP": int(cm_iso[0,1]),
        "TN": int(cm_iso[0,0]), "FN": int(cm_iso[1,0])
    }
}

with open(f"{OUTPUT}/results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("ALL DONE. Results saved to ./outputs/")
print("=" * 60)
print(f"\n{'File':<40} Description")
print("-" * 70)
files_desc = {
    "fig1_class_distribution.png":  "BENIGN vs DDoS class balance",
    "fig2_feature_importance.png":  "Correlation + RF importance side-by-side",
    "fig3_xgb_confusion.png":       "XGBoost confusion matrix",
    "fig4_iso_confusion.png":       "Isolation Forest confusion matrix",
    "fig5_roc_curves.png":          "ROC curves - both models overlaid",
    "fig6_shap_summary_bar.png":    "SHAP global feature importance",
    "fig7_shap_beeswarm.png":       "SHAP direction of influence",
    "fig8_shap_ddos_example.png":   "SHAP waterfall - confirmed DDoS",
    "fig9_shap_fp_example.png":     "SHAP waterfall - false positive",
    "results_summary.json":         "All numeric results for report"
}
for fname, desc in files_desc.items():
    exists = "✓" if os.path.exists(f"{OUTPUT}/{fname}") else "✗"
    print(f"  {exists}  {fname:<38} {desc}")
