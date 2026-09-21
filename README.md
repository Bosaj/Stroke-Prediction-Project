<div align="center">

<!-- Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,3,5,30&height=200&section=header&text=Stroke%20Prediction%20Project&fontSize=32&animation=twinkling&fontAlignY=35&desc=Clinical%20Machine%20Learning%20%7C%20Module%20Machine%20Learning%201%20(ML1)&descSize=14&descAlignY=55" alt="Stroke Prediction Banner" width="100%" />

<!-- Typing Animation -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&repeat=true&width=800&height=40&lines=Machine%20Learning%201%20(ML1)%20Academic%20Project;Clinical%20Stroke%20Risk%20Prediction%20Pipeline;Imbalanced%20Data%20Mitigation%20%26%20SMOTE;XGBoost%20Classifier%20%26%20Interactive%20Streamlit" alt="Typing SVG" />
</p>

<!-- Quick Action Badges -->
<p align="center">
  <a href="https://huggingface.co/spaces/bosaj/clinical-stroke-risk-predictor" target="_blank"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Predictor-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Live Demo" /></a>
  <a href="https://huggingface.co/datasets/bosaj/stroke-prediction-dataset" target="_blank"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-00D9FF?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset" /></a>
  <a href="https://colab.research.google.com/github/Bosaj/Stroke-Prediction-Project/blob/main/notebook/Equipe_NotebookV1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" /></a>
</p>

<!-- Quality & Community Badges -->
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT License" /></a>
  <a href="https://github.com/Bosaj/Stroke-Prediction-Project/actions"><img src="https://img.shields.io/badge/CI%20Pipeline-Passing-brightgreen?style=flat-square&logo=githubactions" alt="CI Status" /></a>
  <a href="https://github.com/Bosaj/Stroke-Prediction-Project/wiki"><img src="https://img.shields.io/badge/Documentation-GitHub%20Wiki-blue.svg?style=flat-square" alt="Wiki" /></a>
  <img src="https://img.shields.io/badge/Module-Machine_Learning_1_(ML1)-00D9FF?style=flat-square&logo=mortarboard&logoColor=white" alt="ML1 Module" />
  <img src="https://img.shields.io/badge/Institution-ENIAD%20Berkane-FF6B00?style=flat-square" alt="ENIAD Berkane" />
</p>

</div>

<!-- Divider -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="Divider" width="100%" />

A machine learning pipeline that predicts a patient's risk of stroke from clinical and demographic attributes, packaged with an interactive Streamlit demo.

## 📖 Academic Course Context — Machine Learning 1 (ML1)

This project was developed within the **Machine Learning 1** engineering module at the **École Nationale d'Intelligence Artificielle et du Digital (ENIAD)**, Mohammed First University, Berkane, Morocco. It demonstrates end-to-end clinical machine learning workflows, advanced data preprocessing, imbalanced data handling, and production model packaging.

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

## 👥 Project Engineering Team (2024)

This project was collaboratively engineered by:
- **EL HADJI Oussama** — Machine Learning Modeling & Interactive Architecture Lead ([GitHub @Bosaj](https://github.com/Bosaj) • [HF @bosaj](https://huggingface.co/bosaj))
- **BAHAYA Radia** — Data Preprocessing, Outlier Remediation & Biomarker Analysis ([HF @Radia1](https://huggingface.co/Radia1) • [LinkedIn](https://www.linkedin.com/in/radia-bahaya-bb6a52260/))
- **GHAFFARI Oualid** — Model Evaluation, ROC/AUC Curves & Diagnostic Metrics ([HF @oualidghaffari](https://huggingface.co/oualidghaffari) • [LinkedIn](https://www.linkedin.com/in/ghaffari-oualid/))
- **SADOG Imane** — Class Imbalance Handling, SMOTE Oversampling & Cross-Validation ([GitHub @ImaneSDG](https://github.com/ImaneSDG) • [LinkedIn](https://www.linkedin.com/in/imane-sadog-a58aa8313/))


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
