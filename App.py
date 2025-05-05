import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

model_rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Air Quality Prediction (PM2.5)")
st.write("Enter the pollutant levels below to predict PM2.5:")

O3 = st.number_input("O3 (Ozone)", min_value=0.0, value=10.0)
NO2 = st.number_input("NO2 (Nitrogen Dioxide)", min_value=0.0, value=10.0)
SO2 = st.number_input("SO2 (Sulfur Dioxide)", min_value=0.0, value=10.0)
CO = st.number_input("CO (Carbon Monoxide)", min_value=0.0, value=1.0)
AQI = st.number_input("AQI (Air Quality Index)", min_value=0.0, value=50.0)

if st.button("Predict PM2.5"):
    input_data = np.array([[O3, NO2, SO2, CO, AQI]])
    input_scaled = scaler.transform(input_data)
    prediction = model_rf.predict(input_scaled)
    st.success(f"Predicted PM2.5: {prediction[0]:.2f} µg/m³")
