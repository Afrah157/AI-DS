%%writefile app.py
import streamlit as st
import joblib
import numpy as np

model = joblib.load("random_forest_aqi_model.joblib")

st.title("AIR QUALITY PREDICTION")

pm25 = st.number_input("PM2.5", value=110.0)
pm10 = st.number_input("PM10", value=135.0)
no2 = st.number_input("NO2", value=45.0)
co = st.number_input("CO", value=1.0)
temp = st.number_input("Temperature", value=28.0)
humidity = st.number_input("Humidity", value=55.0)

if st.button("PREDICT AQI"):
    features = np.array([[pm25, pm10, no2, co, temp, humidity]])
    prediction = model.predict(features)
    st.markdown(f"### Predicted AQI: {prediction[0]:.2f}")
