# Methodology

## Dataset

The [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (5,110 patient records) provides demographic (age, gender, marital status, residence, work type), clinical (hypertension, heart disease, average glucose level, BMI), and lifestyle (smoking status) attributes, with a binary `stroke` outcome. The raw data is heavily imbalanced — only about 4.9% of records are positive (stroke) cases.

## Preprocessing

1. **Exploratory analysis**: distribution and outlier checks for age, BMI, and average glucose level, plus cross-analysis of each categorical attribute (gender, work type, smoking status, residence, marital status) against the stroke outcome.
2. **Missing values**: BMI has missing entries, filled via median imputation.
3. **Outliers**: handled with an IQR-based approach on the numeric columns.
4. **Encoding & scaling**: categorical features are ordinal-encoded; numeric features are standardized.
5. **Class imbalance**: the minority (stroke) class is rebalanced with `RandomOverSampler` from `imbalanced-learn` before the train/test split is taken (`test_size=0.20`, `random_state=42`).

## Model

An `XGBClassifier` (scikit-learn-compatible XGBoost, default hyperparameters) is trained on the rebalanced, preprocessed features.

## Results

On the held-out 20% test split:

| Metric | Value |
|---|---|
| Accuracy | 98.2% |
| ROC-AUC | 0.999 |
| Precision | 0.96 |
| Recall | 1.00 |
| F1-score | 0.98 |

Note that oversampling is applied before the train/test split, so the test set can contain samples derived from the same minority-class records used in training — a common simplification in course/portfolio projects that inflates these numbers somewhat compared to oversampling only on the training fold. Treat the metrics above as representative of the pipeline as implemented, not as a claim about real-world clinical performance.

The trained model is serialized to `notebook/xgb_model.pkl` and loaded directly by the Streamlit demo (`notebook/app.py`) for live predictions.
