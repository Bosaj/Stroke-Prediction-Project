# Stroke Risk Prediction ML System 🏥

[![CI & Observability](https://img.shields.io/badge/CI%2FCD-Passing-success?logo=githubactions&logoColor=white)](https://github.com/Bosaj/Stroke-Prediction-Project/actions)
[![SLSA Attestation](https://img.shields.io/badge/SLSA%20Level%203-Attested-blue?logo=githubactions&logoColor=white)](https://github.com/Bosaj/Stroke-Prediction-Project/attestations)
[![GHCR Container](https://img.shields.io/badge/GHCR-ghcr.io%2Fbosaj%2Fstroke-prediction-project-brightgreen?logo=docker&logoColor=white)](https://github.com/Bosaj?tab=packages)
[![Project Roadmap](https://img.shields.io/badge/Project%20Roadmap-%2322-8A2BE2?logo=github&logoColor=white)](https://github.com/users/Bosaj/projects/22)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Clinical%20Risk%20Calculator-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

An end-to-end Machine Learning healthcare platform engineered to predict stroke risk probability based on clinical biomarkers, demographic attributes, and lifestyle parameters using an optimized **XGBoost Classifier**.

---

## 🌟 Interactive Streamlit Clinical App

Run the interactive clinical prediction interface:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run streamlit_app.py
```

### Features & Clinical Inputs:
- **Biometric Factors**: Age, Body Mass Index (BMI), Average Blood Glucose Level.
- **Cardiovascular History**: Hypertension status, Pre-existing Heart Disease.
- **Lifestyle & Demographic Attributes**: Smoking status (formerly, never, current), Work environment, Residence type (urban/rural), Marital status.
- **Real-Time Risk Stratification**: Instant inference with calibrated probability output, risk category grading, and preventive clinical guidance.

---

## 🔬 Model Architecture & Training

The predictive engine is trained on validated clinical stroke datasets:
1. **Data Cleaning & Handling Imbalance**: Handled extreme class skew using advanced resampling and stratified cross-validation.
2. **Feature Encoding**: Exact ordinal encoding preserved across training and deployment pipelines (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`).
3. **Optimized XGBoost Model (`xgb_model.pkl`)**: Gradient boosted trees tuned with hyperparameter optimization for high recall on positive risk cases.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Bosaj/Stroke-Prediction-Project.git
cd Stroke-Prediction-Project

# Virtual environment setup
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 👤 Author

**Oussama EL HADJI (Bosaj)**
- GitHub: [@Bosaj](https://github.com/Bosaj)
- Institution: **ENIAD — École Nationale d'Intelligence Artificielle et du Digital**