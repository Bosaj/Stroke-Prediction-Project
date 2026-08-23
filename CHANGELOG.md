# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-16

### Added
- Exploratory data analysis notebook (`notebook/Equipe_NotebookV1.ipynb`) covering the 5,110-record stroke prediction dataset: missing-value analysis, outlier detection (BMI, average glucose level), categorical distribution analysis, and cross-analysis against the stroke outcome.
- Data preprocessing: median imputation for BMI, IQR-based outlier handling, ordinal encoding, standard scaling, and random oversampling of the minority class.
- XGBoost classifier achieving 98.2% accuracy and 0.999 ROC-AUC on the held-out test set.
- Streamlit demo application (`notebook/app.py`) for interactive stroke-risk prediction using the trained model (`xgb_model.pkl`).
- `README.md`, `LICENSE` (MIT), `requirements.txt`, and this changelog.
- `.github/workflows/ci.yml`: GitHub Actions pipeline validating notebook integrity and linting the Streamlit app.
