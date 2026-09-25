from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# Load the pre-trained model (path resolved relative to this file so it
# works regardless of the working directory the app is launched from).
MODEL_PATH = Path(__file__).parent / "xgb_model.pkl"
model = joblib.load(MODEL_PATH)

# Category -> integer mappings, reconstructed from the exact order
# `datav1[col].unique()` produced on the real training data in the
# notebook's OrdinalEncoder step (see notebook/Equipe_NotebookV1.ipynb).
# These must match training exactly - a LabelEncoder fit fresh on a
# single input value (the previous approach) always maps that value to
# 0, silently discarding every categorical input.
GENDER_MAP = {"Male": 0, "Female": 1}
EVER_MARRIED_MAP = {"Yes": 0, "No": 1}
WORK_TYPE_MAP = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "children": 3,
    "Never_worked": 4,
}
RESIDENCE_TYPE_MAP = {"Urban": 0, "Rural": 1}
SMOKING_STATUS_MAP = {
    "formerly smoked": 0,
    "never smoked": 1,
    "smokes": 2,
    "Unknown": 3,
}


def predict(
    age,
    avg_glucose_level,
    bmi,
    gender,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    smoking_status,
):
    # Column order must match training exactly: gender, age, hypertension,
    # heart_disease, ever_married, work_type, Residence_type,
    # avg_glucose_level, bmi, smoking_status.
    input_data = np.array(
        [
            [
                GENDER_MAP[gender],
                age,
                hypertension,
                heart_disease,
                EVER_MARRIED_MAP[ever_married],
                WORK_TYPE_MAP[work_type],
                RESIDENCE_TYPE_MAP[residence_type],
                avg_glucose_level,
                bmi,
                SMOKING_STATUS_MAP[smoking_status],
            ]
        ]
    )

    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[:, 1]
    return prediction[0], prediction_prob[0]


st.title("Stroke Risk Prediction (XGBoost model)")
st.caption(
    "Trained on the Stroke Prediction Dataset (5,110 records). "
    "For demonstration purposes only - not a medical device."
)

age = st.number_input("Age", min_value=1, max_value=120, value=67)
avg_glucose_level = st.number_input(
    "Average glucose level", min_value=0.0, value=228.69
)
bmi = st.number_input("BMI", min_value=0.0, value=36.6)

gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart disease", [0, 1])
ever_married = st.selectbox("Ever married", list(EVER_MARRIED_MAP.keys()))
work_type = st.selectbox("Work type", list(WORK_TYPE_MAP.keys()))
residence_type = st.selectbox("Residence type", list(RESIDENCE_TYPE_MAP.keys()))
smoking_status = st.selectbox("Smoking status", list(SMOKING_STATUS_MAP.keys()))

if st.button("Predict"):
    prediction, prediction_prob = predict(
        age,
        avg_glucose_level,
        bmi,
        gender,
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        residence_type,
        smoking_status,
    )

    if prediction == 1:
        st.error(f"High stroke risk predicted (probability: {prediction_prob:.2f})")
    else:
        st.success(f"Low stroke risk predicted (probability: {prediction_prob:.2f})")
