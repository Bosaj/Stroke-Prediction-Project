# Stroke-Prediction-Project

![CI Pipeline](https://github.com/Bosaj/Stroke-Prediction-Project/actions/workflows/ci_qa_monitoring.yml/badge.svg)
[![GitHub Wiki](https://img.shields.io/badge/Documentation-GitHub%20Wiki-blue.svg)](https://github.com/Bosaj/Stroke-Prediction-Project/wiki)
[![Quality Gate](https://img.shields.io/badge/Quality%20Gate-Passed-brightgreen.svg)](docs/MONITORING_AND_QA.md)

---

![CI](https://github.com/Bosaj/Stroke-Prediction-Project/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)

A machine learning pipeline that predicts a patient's risk of stroke from clinical and demographic attributes, packaged with an interactive Streamlit demo.

## Overview

Using the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (5,110 patient records), this project performs a full exploratory data analysis, handles missing values and outliers, corrects for severe class imbalance (only ~4.9% of patients had a stroke), and trains an XGBoost classifier to flag at-risk patients. A Streamlit app lets you enter a patient's profile and get a live prediction from the trained model.

## Features

- **Exploratory Data Analysis**: distribution and outlier analysis for age, BMI, average glucose level, and categorical attributes (gender, work type, smoking status, residence, marital status), plus cross-analysis against the stroke outcome.
- **Data cleaning**: median imputation for missing BMI values, IQR-based outlier handling, ordinal encoding of categorical features, standard scaling of numeric features.
- **Class imbalance handling**: random oversampling of the minority (stroke) class via `imbalanced-learn`.
- **Model**: `XGBClassifier` reaching **98.2% accuracy** and a **0.999 ROC-AUC** on the held-out test split (precision 0.96, recall 1.00, F1 0.98).
- **Interactive demo**: a Streamlit app (`notebook/app.py`) that loads the trained model (`xgb_model.pkl`) and predicts stroke risk from user-entered patient data.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, Matplotlib/Seaborn, Streamlit.

## Getting Started

### Prerequisites
- Python 3.10+

### Installation
```bash
pip install -r requirements.txt
```

### Usage

Explore the analysis and training pipeline:
```bash
jupyter notebook notebook/Equipe_NotebookV1.ipynb
```

Run the interactive prediction demo:
```bash
cd notebook
streamlit run app.py
```

## Testing / CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) validates the notebook's structural integrity, installs the full dependency set, and lints the Streamlit app for critical errors on every push.

## Project Structure

```
Stroke-Prediction-Project/
├── data/
│   └── stroke_prediction.csv       # Source dataset (5,110 records)
├── notebook/
│   ├── Equipe_NotebookV1.ipynb     # EDA, preprocessing, and model training
│   ├── app.py                      # Streamlit prediction demo
│   └── xgb_model.pkl               # Trained XGBoost model
└── requirements.txt
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Author

Oussama EL HADJI — [github.com/Bosaj](https://github.com/Bosaj)


## 📊 Monitoring, Controlling, Evaluation & QA

This project includes a standardized 4-Pillar Observability and QA framework:
- **Logs & Prometheus/Grafana Monitoring**: Configured in `monitoring/` with Prometheus scraper configs and Grafana dashboards.
- **Health Controlling & Evaluation**: Liveness/readiness controllers in `monitoring/health.py` and evaluation harness in `scripts/eval_harness.py`.
- **QA & Testing**: Automated Pytest/Vitest integration and CI workflows via `.github/workflows/ci_qa_monitoring.yml`.

For complete instructions, architecture details, and commands, see [docs/MONITORING_AND_QA.md](docs/MONITORING_AND_QA.md).

---

## 📚 Documentation & GitHub Wiki
- 📖 **Official Project Wiki**: [https://github.com/Bosaj/Stroke-Prediction-Project/wiki](https://github.com/Bosaj/Stroke-Prediction-Project/wiki)
- 🔍 **Architecture & Design**: [https://github.com/Bosaj/Stroke-Prediction-Project/wiki/Architecture-and-Design](https://github.com/Bosaj/Stroke-Prediction-Project/wiki/Architecture-and-Design)
- 🚀 **Getting Started Guide**: [https://github.com/Bosaj/Stroke-Prediction-Project/wiki/Getting-Started](https://github.com/Bosaj/Stroke-Prediction-Project/wiki/Getting-Started)
- 📊 **Monitoring & Observability**: [docs/MONITORING_AND_QA.md](docs/MONITORING_AND_QA.md)
