import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras.models import load_model

from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained neural network model and scaler
# model = load_model("stress_model1.h5")  # You must save this from your training code
model = tf.keras.models.load_model("stress_model.h5")
scaler = joblib.load("scaler.pkl")     # Save the StandardScaler after fitting

# Page Title
st.title("🧠 Cognifit: Stress Level Predictor")
st.write("This app predicts your mental health stress level based on lifestyle and health factors.")

# User Inputs
anxiety = st.slider("Anxiety Level", 0, 20, 10)
mental_health_history = st.selectbox("Any Past Mental Health Condition?", [0,1])
depression = st.slider("Depression Level", 0, 30, 10)
headache = st.slider("Headache Frequency", 0, 10, 5)
sleep_quality = st.slider("Sleep Quality (1-Poor to 5-Great)", 0, 3, 10)
breathing_problem = st.slider("Breathing Difficulty (0-No to 5-Severe)", 0, 5, 2)
living_conditions = st.slider("Living Conditions (1-Worst to 5-Best)", 0, 5, 3)
academic_performance = st.slider("Academic Performance (1-Low to 5-High)", 0, 5, 3)
study_load = st.slider("Study Load (1-Low to 5-High)", 0, 5, 3)
career_concerns = st.slider("Future Career Concerns (1-Low to 5-High)", 0, 5, 3)
extracurricular = st.slider("Extracurricular Activity Level (1-Low to 5-High)", 0, 5, 3)

#mental_health_history= 1 if 'Yes' else 0
# Collect inputs into a single array
input_data = np.array([[anxiety, mental_health_history, depression, headache, sleep_quality,
                        breathing_problem, living_conditions, academic_performance,
                        study_load, career_concerns, extracurricular]])

# Predict button
if st.button("Predict Stress Level"):
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)

    # Display result
    stress_labels = {0: "Low", 1: "Moderate", 2: "High"}
    st.success(f"🧘 Your Predicted Stress Level: **{stress_labels[predicted_class]}**")
