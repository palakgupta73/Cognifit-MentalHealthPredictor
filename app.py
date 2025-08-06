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
mental_health_history = st.radio("🧠 Past Mental Health Condition?", ["No", "Yes"])
anxiety = st.slider("😰 Anxiety Level", 0, 20, 10)
mental_health_history = 1 if mental_health_history == "Yes" else 0
depression = st.slider("😞 Depression Level", 0, 30, 10)
headache = st.slider("🤕 Headache Frequency", 0, 5,2)
breathing_problem = st.slider("😮‍💨 Breathing Difficulty", 0, 5, 2)
sleep_quality = st.slider("🛌 Sleep Quality", 0, 5, 2)
living_conditions = st.slider("🏠 Living Conditions", 0, 5, 2)
academic_performance = st.slider("📚 Academic Performance", 0, 5, 2)
study_load = st.slider("📖 Work Load", 0, 5, 2)
career_concerns = st.slider("💼 Career Concerns", 0, 5, 2)
extracurricular = st.slider("🎭 Extracurricular Activity", 0, 5, 2)

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

    stress_suggestions = {
        0: "✅ Keep up the good work! Continue maintaining a healthy balance of study, rest, and personal time.",
        1: "☘️ You're doing okay, but consider incorporating short breaks, mindfulness, or talking to someone you trust to reduce stress.",
        2: "⚠️ Your stress level seems high. Please consider seeking professional help, improving sleep, or adjusting workload for better well-being."
    }
    # Display result
    colors = {0: "🟢", 1: "🟠", 2: "🔴"}
    emoji = colors[predicted_class]
    stress_labels = {0: "Low", 1: "Moderate", 2: "High"}
    level = stress_labels[predicted_class]
    st.success(f"{emoji} Your Predicted Stress Level: *{level}*")
    suggestion_output=stress_suggestions[predicted_class]
    st.info(suggestion_output)
    download_text = (
        f"🧠 Predicted Stress Level: {level}\n\n"
        f"📋 Suggestion:\n{suggestion_output}\n\n"
        f"🧾 Your Responses:\n"
        f"• Past Mental Health History: {'Yes' if mental_health_history == 1 else 'No'}\n"
        f"• Anxiety Level: {anxiety}/20\n"
        f"• Depression Level: {depression}/30\n"
        f"• Headache Frequency: {headache}/5\n"
        f"• Breathing Difficulty: {breathing_problem}/5\n"
        f"• Sleep Quality: {sleep_quality}/5\n"
        f"• Living Conditions: {living_conditions}/5\n"
        f"• Academic Performance: {academic_performance}/5\n"
        f"• Work Load: {study_load}/5\n"
        f"• Career Concerns: {career_concerns}/5\n"
        f"• Extracurricular Activity: {extracurricular}/5\n"
    )
    # Optional download
    # download_text = f"Predicted Stress Level: {level}\n\nSuggestion:\n{suggestion_output}\n\nYour Response was:\n{input_data}"
    st.download_button("💾 Download Result", download_text, file_name="prediction.txt")