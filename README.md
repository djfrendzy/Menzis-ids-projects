# 🛡️ AI-Driven Network Intrusion Detection — Menzis Finance

> **Coursework Project | Usable Cyber Security Module**  
> Applying machine learning to detect DDoS attacks on real enterprise network traffic.

---

## 📌 Overview

This repository contains two end-to-end ML pipelines for detecting DDoS network intrusions, built using the **CIC-IDS-2017 benchmark dataset** (Friday Afternoon — 225,745 real network flows). Both projects were developed as part of a university cybersecurity module, simulating the role of a data scientist working for a fictional global bank, Menzis Finance.

Each project addresses four SOC (Security Operations Centre) requirements:
- ✅ High-precision detection — outperforming rule-based/signature systems
- ✅ Explainable alerts — using SHAP so analysts understand *why* an alert fired
- ✅ Minimal false positives — reducing analyst alert fatigue
- ✅ Novel attack detection — unsupervised models catch zero-day threats with no labels

---

## 📁 Repository Structure

```
menzis-ids-project/
│
├── project1_xgboost_isolation_forest/   # Project 1: XGBoost + Isolation Forest
│   ├── ids_pipeline.py                  # Full ML pipeline
│   ├── requirements.txt                 # Python dependencies
│   ├── README.md                        # Project-specific documentation
│   └── outputs/                         # Generated figures (gitignored if large)
│
├── project2_randomforest_lof/           # Project 2: Random Forest + LOF
│   ├── ids_rf_lof_pipeline.py           # Full ML pipeline
│   ├── requirements.txt                 # Python dependencies
│   ├── README.md                        # Project-specific documentation
│   └── outputs/                         # Generated figures
│
├── .gitignore                           # Excludes data files and outputs
├── LICENSE                              # MIT licence
└── README.md                            # This file
```

---

## 🔬 Projects at a Glance

### Project 1 — XGBoost + Isolation Forest
| | |
|---|---|
| **Supervised Model** | XGBoost (Extreme Gradient Boosting) |
| **Unsupervised Model** | Isolation Forest |
| **Explainability** | SHAP TreeExplainer |
| **Precision** | 0.9999 |
| **F1 Score** | 0.9999 |
| **ROC-AUC** | 1.0000 |

### Project 2 — Random Forest + Local Outlier Factor
| | |
|---|---|
| **Supervised Model** | Random Forest |
| **Unsupervised Model** | Local Outlier Factor (LOF) |
| **Explainability** | SHAP TreeExplainer |
| **Precision** | 0.9998 |
| **F1 Score** | 0.9999 |
| **ROC-AUC (LOF)** | 0.9187 |

---

## 📊 Dataset

**CIC-IDS-2017 — Friday Working Hours (Afternoon) — DDoS**  
Published by the Canadian Institute for Cybersecurity.

| Class | Flows | Share |
|---|---|---|
| DDoS | 128,027 | 56.7% |
| BENIGN | 97,718 | 43.3% |
| **Total** | **225,745** | |

> ⚠️ **The dataset CSV is not included in this repo** (file size ~50MB). Download it from:  
> https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV/

Once downloaded, update `CSV_PATH` at the top of each pipeline script to point to your local copy.

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/menzis-ids-project.git
cd menzis-ids-project
```

### 2. Install dependencies
```bash
# For Project 1
cd project1_xgboost_isolation_forest
pip install -r requirements.txt

# For Project 2
cd ../project2_randomforest_lof
pip install -r requirements.txt
```

### 3. Set your dataset path
Open the pipeline script and update line:
```python
CSV_PATH = r"C:\path\to\Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv"
```

### 4. Run
```bash
python ids_pipeline.py          # Project 1
python ids_rf_lof_pipeline.py   # Project 2
```
All figures and results are saved to the `outputs/` folder automatically.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![SHAP](https://img.shields.io/badge/SHAP-0.44-red)
![pandas](https://img.shields.io/badge/pandas-2.0-lightblue?logo=pandas)

- **Language:** Python 3.9+
- **ML Libraries:** scikit-learn, XGBoost, SHAP
- **Data:** pandas, numpy
- **Visualisation:** matplotlib, seaborn

---

## 📄 Reports

Full project reports (IEEE referenced, with embedded figures) are available as Word documents in each project folder.

---

## 👤 Author

**[Your Full Name]**  
MSc [Your Programme] — [Your University]  
[LinkedIn Profile URL]  
[Your Email — optional]

---

## 📜 Licence

This project is licensed under the MIT Licence — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset: Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.
- SHAP: Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
