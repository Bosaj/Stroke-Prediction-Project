# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `notebook/app.py` built its feature array in the wrong column order and re-fit a throwaway `LabelEncoder` on each single input value (always encoding to 0), so every categorical field was silently ignored and numeric fields were fed into the wrong model input slots. Rebuilt the categorical mappings from the training notebook's actual encoding order and corrected the column order to match training. See the commit message for full detail.

### Added
- `.github/workflows/pages.yml`: publishes the notebook's rendered outputs as a static GitHub Pages site on every push to `main`.

## [1.0.0] - 2024-12-16

### Added
- Exploratory data analysis notebook (`notebook/Equipe_NotebookV1.ipynb`) covering the 5,110-record stroke prediction dataset: missing-value analysis, outlier detection (BMI, average glucose level), categorical distribution analysis, and cross-analysis against the stroke outcome.
- Data preprocessing: median imputation for BMI, IQR-based outlier handling, ordinal encoding, standard scaling, and random oversampling of the minority class.
- XGBoost classifier achieving 98.2% accuracy and 0.999 ROC-AUC on the held-out test set.
- Streamlit demo application (`notebook/app.py`) for interactive stroke-risk prediction using the trained model (`xgb_model.pkl`).
- `README.md`, `LICENSE` (MIT), `requirements.txt`, and this changelog.
- `.github/workflows/ci.yml`: GitHub Actions pipeline validating notebook integrity and linting the Streamlit app.
