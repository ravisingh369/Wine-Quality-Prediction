import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("wine_model.pkl")

st.set_page_config(page_title="Wine Quality Detector", page_icon="🍷")

st.title("🍷 Wine Quality Detector")
st.write("Enter wine chemical properties to predict quality")

fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
ph = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

if st.button("Predict Wine Quality"):
    input_data = np.array([
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]).reshape(1, -1)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🍷 Mast Daaru hai (Good Quality Wine)")
    else:
        st.error("☠️ Zaher wali hai bhai (Low Quality Wine)")
