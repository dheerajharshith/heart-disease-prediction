import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and feature names
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("model_columns.pkl")

st.title("💓 Heart Disease Prediction App")

# Collect user input
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 50, 250)
cholesterol = st.number_input("Cholesterol", 100, 600)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate Achieved", 60, 220)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Manual encodings (based on LabelEncoder from training)
sex_map = {"Male": 1, "Female": 0}
cp_map = {"TA": 3, "ATA": 2, "NAP": 1, "ASY": 0}
ecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
angina_map = {"Yes": 1, "No": 0}
slope_map = {"Up": 2, "Flat": 1, "Down": 0}

# Prepare input vector
input_data = np.array([[
    age,
    sex_map[sex],
    cp_map[chest_pain],
    resting_bp,
    cholesterol,
    fasting_bs,
    ecg_map[resting_ecg],
    max_hr,
    angina_map[exercise_angina],
    oldpeak,
    slope_map[st_slope]
]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")