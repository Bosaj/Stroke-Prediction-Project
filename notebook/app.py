import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Charger le modèle préalablement sauvegardé
model = joblib.load('xgb_model.pkl')

# Fonction pour encoder les variables catégorielles
def encode_input_data(gender, ever_married, work_type, residence_type, smoking_status):
    # Créer un encodeur pour chaque variable catégorielle
    label_encoder = LabelEncoder()
    
    # Encoder chaque variable catégorielle
    gender_encoded = label_encoder.fit_transform([gender])[0]
    ever_married_encoded = label_encoder.fit_transform([ever_married])[0]
    work_type_encoded = label_encoder.fit_transform([work_type])[0]
    residence_type_encoded = label_encoder.fit_transform([residence_type])[0]
    smoking_status_encoded = label_encoder.fit_transform([smoking_status])[0]
    
    return gender_encoded, ever_married_encoded, work_type_encoded, residence_type_encoded, smoking_status_encoded

# Fonction pour prédire à partir des entrées
def predict(age, avg_glucose_level, bmi, gender, hypertension, heart_disease, ever_married, work_type, residence_type, smoking_status):
    # Encoder les variables catégorielles
    gender_encoded, ever_married_encoded, work_type_encoded, residence_type_encoded, smoking_status_encoded = encode_input_data(gender, ever_married, work_type, residence_type, smoking_status)

    # Créer un tableau numpy avec les valeurs saisies
    input_data = np.array([[age, avg_glucose_level, bmi, gender_encoded, hypertension, heart_disease, ever_married_encoded, work_type_encoded, residence_type_encoded, smoking_status_encoded]])

    # Effectuer la prédiction avec le modèle chargé
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[:, 1]
    
    return prediction[0], prediction_prob[0]

# Titre de l'application
st.title('Prédiction de la maladie (Modèle XGBoost)')

# Collecte des données via les widgets Streamlit
age = st.number_input('Âge', min_value=1, max_value=120, value=67)
avg_glucose_level = st.number_input('Niveau moyen de glucose', min_value=0.0, value=228.69)
bmi = st.number_input('IMC', min_value=0.0, value=36.6)

gender = st.selectbox('Sexe', ['Male', 'Female'])
hypertension = st.selectbox('Hypertension', [0, 1])  # 0: Non, 1: Oui
heart_disease = st.selectbox('Maladie cardiaque', [0, 1])  # 0: Non, 1: Oui
ever_married = st.selectbox('Marié(e) ou non', ['Yes', 'No'])
work_type = st.selectbox('Type de travail', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Type de résidence', ['Urban', 'Rural'])
smoking_status = st.selectbox('Statut de fumeur', ['never smoked', 'formerly smoked', 'smokes'])

# Bouton pour prédire
if st.button('Prédire'):
    # Appeler la fonction de prédiction
    prediction, prediction_prob = predict(age, avg_glucose_level, bmi, gender, hypertension, heart_disease, ever_married, work_type, residence_type, smoking_status)
    
    # Afficher le résultat
    if prediction == 1:
        st.write(f'Le modèle prédit que vous avez une forte probabilité de maladie (Probabilité: {prediction_prob:.2f})')
    else:
        st.write(f'Le modèle prédit que vous n\'avez pas de maladie (Probabilité: {prediction_prob:.2f})')
