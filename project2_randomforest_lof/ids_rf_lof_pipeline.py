import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, json
warnings.filterwarnings('ignore')

# CONFIGURATION
CSV_PATH = "Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv"
OUTPUT   = "outputs_v2"
os.makedirs(OUTPUT, exist_ok=True)

# SECTION 1 - Load & Explore
print("SECTION 1 - Data Exploration & Preparation")
print("-" * 40)

df = pd.read_csv(CSV_PATH, encoding='utf-8', low_memory=False)
df.columns = df.columns.str.strip()

print(f"Shape          : {df.shape}")
print(f"\nClass distribution:\n{df['Label'].value_counts()}")
print(f"\nClass balance (%):\n{df['Label'].value_counts(normalize=True).mul(100).round(2)}")

# Replace infinities with NaN, then fill with median
df.replace([np.inf, -np.inf], np.nan, inplace=True)
missing = df.isnull().sum()
print(f"\nMissing values before cleaning:\n{missing[missing > 0]}")
df.fillna(df.median(numeric_only=True), inplace=True)
print("Filled missing values with column medians.")

# Binary label: BENIGN=0, DDoS=1
df['binary_label'] = (df['Label'] == 'DDoS').astype(int)

# Class distribution chart
fig, ax = plt.subplots(figsize=(7, 4))
counts = df['Label'].value_counts()
counts.plot(kind='bar', ax=ax, color=['#1565C0', '#C62828'], edgecolor='black', width=0.5)
ax.set_title('Class Distribution – CIC-IDS-2017 DDoS Dataset', fontsize=13, fontweight='bold')
ax.set_xlabel('Traffic Class'); ax.set_ylabel('Number of Flows')
ax.tick_params(axis='x', rotation=0)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig1_class_distribution.png", dpi=150)
plt.close()
print("Saved  fig1_class_distribution.png")


# SECTION 2 - Feature Engineering & Selection
print("SECTION 2 - Feature Engineering & Selection")
print("-" * 60)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [c for c in df.columns if c not in ['Label', 'binary_label']]
X = df[FEATURE_COLS]
y = df['binary_label']

# Stratified 70/15/15 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

# Scale - fit ONLY on train to prevent leakage
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
X_val_sc   = pd.DataFrame(scaler.transform(X_val),       columns=FEATURE_COLS)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS)

y_train = y_train.reset_index(drop=True)
y_val   = y_val.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# --- Technique 1: Pearson Correlation ---
corr = X_train_sc.corrwith(y_train).abs().sort_values(ascending=False)
print(f"\nTop 10 by Pearson Correlation:\n{corr.head(10).to_string()}")

# --- Technique 2: Random Forest importance (quick shallow trees) ---
from sklearn.ensemble import RandomForestClassifier

print("\nFitting feature-selection Random Forest...")
rf_sel = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
rf_sel.fit(X_train_sc, y_train)
rf_imp = pd.Series(rf_sel.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\nTop 10 by RF Gini Importance:\n{rf_imp.head(10).to_string()}")

# Importance plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
corr.head(15).sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Feature Importance - Pearson Correlation', fontweight='bold')
axes[0].set_xlabel('|Pearson r| with DDoS label')

rf_imp.head(15).sort_values().plot(kind='barh', ax=axes[1], color='darkorange')
axes[1].set_title('Feature Importance - Random Forest Gini', fontweight='bold')
axes[1].set_xlabel('Mean Decrease in Impurity')

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig2_feature_importance.png", dpi=150)
plt.close()
print("Saved  fig2_feature_importance.png")

# Union of top 15 from both methods
SELECTED = sorted(list(set(corr.head(15).index) | set(rf_imp.head(15).index)))
print(f"\nSelected {len(SELECTED)} features.")

X_train_sel = X_train_sc[SELECTED].reset_index(drop=True)
X_val_sel   = X_val_sc[SELECTED].reset_index(drop=True)
X_test_sel  = X_test_sc[SELECTED].reset_index(drop=True)

# SECTION 3 - Model Development
print("SECTION 3 - Model Development")
print("-" * 60)

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support,
    ConfusionMatrixDisplay, roc_curve
)

#MODEL A: Random Forest (Supervised)
print("\n Model A: Random Forest (Supervised)")

rf_model = RandomForestClassifier(
    n_estimators=300,       # 300 trees for stable predictions
    max_depth=20,           # deep enough to capture complex patterns
    min_samples_leaf=2,     # prevents overfitting on tiny leaves
    class_weight='balanced',# automatically handles class imbalance
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train_sel, y_train)

y_pred_rf = rf_model.predict(X_test_sel)
y_prob_rf = rf_model.predict_proba(X_test_sel)[:, 1]

prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='binary')
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"\n  Precision : {prec_rf:.4f}")
print(f"  Recall    : {rec_rf:.4f}")
print(f"  F1 Score  : {f1_rf:.4f}")
print(f"  ROC-AUC   : {auc_rf:.4f}")
print(f"\n{classification_report(y_test, y_pred_rf, target_names=['BENIGN','DDoS'])}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, display_labels=['BENIGN', 'DDoS'], cmap='Blues', ax=ax)
ax.set_title('Random Forest - Confusion Matrix (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig3_rf_confusion.png", dpi=150)
plt.close()
print("Saved → fig3_rf_confusion.png")

# MODEL B: Local Outlier Factor (Unsupervised) 
print("\n Model B: Local Outlier Factor (Unsupervised)")

from sklearn.neighbors import LocalOutlierFactor

# Train ONLY on BENIGN flows - LOF learns what normal looks like
X_train_benign = X_train_sel[y_train == 0].reset_index(drop=True)
print(f"  Training LOF on {len(X_train_benign):,} BENIGN flows only...")

# novelty=True allows LOF to score unseen test data
lof_model = LocalOutlierFactor(
    n_neighbors=20,     # compare each flow against its 20 nearest neighbours
    contamination=0.05, # expected anomaly rate in live traffic
    novelty=True,       # required to predict on new data
    n_jobs=-1
)
lof_model.fit(X_train_benign)

# LOF returns +1 (normal) / -1 (outlier) - convert to 0/1
lof_raw    = lof_model.predict(X_test_sel)
y_pred_lof = np.where(lof_raw == -1, 1, 0)

prec_lof, rec_lof, f1_lof, _ = precision_recall_fscore_support(y_test, y_pred_lof, average='binary')
# Score: more negative = more anomalous > flip sign for AUC
scores_lof = -lof_model.decision_function(X_test_sel)
auc_lof    = roc_auc_score(y_test, scores_lof)

print(f"\n  Precision : {prec_lof:.4f}")
print(f"  Recall    : {rec_lof:.4f}")
print(f"  F1 Score  : {f1_lof:.4f}")
print(f"  ROC-AUC   : {auc_lof:.4f}")
print(f"\n{classification_report(y_test, y_pred_lof, target_names=['BENIGN','DDoS'])}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lof, display_labels=['BENIGN', 'DDoS'], cmap='Oranges', ax=ax)
ax.set_title('Local Outlier Factor - Confusion Matrix (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig4_lof_confusion.png", dpi=150)
plt.close()
print("Saved > fig4_lof_confusion.png")

# ROC Curves 
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
fpr_lof, tpr_lof, _ = roc_curve(y_test, scores_lof)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_rf,  tpr_rf,  color='steelblue',  lw=2, label=f'Random Forest  (AUC = {auc_rf:.4f})')
ax.plot(fpr_lof, tpr_lof, color='darkorange', lw=2, label=f'LOF            (AUC = {auc_lof:.4f})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curves - Random Forest vs Local Outlier Factor', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig5_roc_curves.png", dpi=150)
plt.close()
print("Saved  fig5_roc_curves.png")

# SECTION 4 - SHAP Explainability
print("SECTION 4 - SHAP Explainability")
print("-" * 60)

import shap

# Sample 500 test flows for SHAP (faster)
X_shap = X_test_sel.sample(500, random_state=42).reset_index(drop=True)
y_shap = y_test.iloc[X_test_sel.sample(500, random_state=42).index].reset_index(drop=True)

print("Computing SHAP values (this may take 1-2 minutes)...")
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_shap)

# RF returns list [class0_array, class1_array] - take DDoS (class 1)
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# Global bar chart
plt.figure()
shap.summary_plot(sv, X_shap, plot_type='bar', show=False, max_display=15, color='#1976D2')
plt.title('SHAP Feature Importance – Random Forest (Global)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig6_shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved > fig6_shap_bar.png")

# Beeswarm
plt.figure()
shap.summary_plot(sv, X_shap, show=False, max_display=15)
plt.title('SHAP Beeswarm - Feature Impact Direction', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig7_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved > fig7_shap_beeswarm.png")

# Waterfall - confirmed DDoS (true positive)
tp_mask = (y_shap == 1) & (rf_model.predict(X_shap) == 1)
# Handle both single and multi-output SHAP formats
ev = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
# sv shape can be (n, features) or (n, features, 2) - take class-1 slice
if sv.ndim == 3:
    sv_ddos = sv[:, :, 1]
else:
    sv_ddos = sv

if tp_mask.any():
    ex1 = tp_mask.idxmax()
    shap.waterfall_plot(
        shap.Explanation(values=sv_ddos[ex1], base_values=ev,
                         data=X_shap.iloc[ex1], feature_names=list(X_shap.columns)),
        show=False, max_display=12)
    plt.title('SHAP Waterfall - Example 1: Confirmed DDoS Alert', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/fig8_shap_ddos.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved > fig8_shap_ddos.png")

fp_mask = (y_shap == 0) & (rf_model.predict(X_shap) == 1)
if fp_mask.any():
    ex2 = fp_mask.idxmax()
    shap.waterfall_plot(
        shap.Explanation(values=sv_ddos[ex2], base_values=ev,
                         data=X_shap.iloc[ex2], feature_names=list(X_shap.columns)),
        show=False, max_display=12)
    plt.title('SHAP Waterfall - Example 2: False Positive Investigation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/fig9_shap_fp.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved > fig9_shap_fp.png")
else:
    print("No false positives in SHAP sample - RF is extremely precise on this data.")


# SECTION 5 - Save Results
cm_rf  = confusion_matrix(y_test, y_pred_rf)
cm_lof = confusion_matrix(y_test, y_pred_lof)

results = {
    "dataset": {
        "total_flows": int(len(df)),
        "benign": int((df['Label']=='BENIGN').sum()),
        "ddos":   int((df['Label']=='DDoS').sum()),
        "features_total": len(FEATURE_COLS),
        "features_selected": len(SELECTED),
    },
    "splits": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
    "random_forest": {
        "precision": round(prec_rf, 4), "recall": round(rec_rf, 4),
        "f1": round(f1_rf, 4), "roc_auc": round(auc_rf, 4),
        "TP": int(cm_rf[1,1]), "FP": int(cm_rf[0,1]),
        "TN": int(cm_rf[0,0]), "FN": int(cm_rf[1,0])
    },
    "lof": {
        "precision": round(prec_lof, 4), "recall": round(rec_lof, 4),
        "f1": round(f1_lof, 4), "roc_auc": round(auc_lof, 4),
        "TP": int(cm_lof[1,1]), "FP": int(cm_lof[0,1]),
        "TN": int(cm_lof[0,0]), "FN": int(cm_lof[1,0])
    }
}
with open(f"{OUTPUT}/results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("ALL DONE — outputs saved to ./" + OUTPUT + "/")
print("=" * 60)
print(f"\nOutputs folder location:")
import os; print(f"  {os.path.abspath(OUTPUT)}")
print("\nFiles generated:")
for fname in sorted(os.listdir(OUTPUT)):
    print(f"  • {fname}")
