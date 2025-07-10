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
    # Optional download
    download_text = f"Predicted Stress Level: {level}\n\nSuggestion:\n{suggestion_output}"
    st.download_button("💾 Download Result", download_text, file_name="prediction.txt")
    
    # st.success(f"🧘 Your Predicted Stress Level: **{stress_labels[predicted_class]}**")
    
# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import joblib
# import matplotlib.pyplot as plt

# # ------------------- Load Model and Scaler -------------------
# @st.cache_resource
# def load_assets():
#     model = tf.keras.models.load_model("stress_model.h5")
#     scaler = joblib.load("scaler.pkl")
#     return model, scaler

# model, scaler = load_assets()

# # ------------------- App Title -------------------
# # st.set_page_config(page_title="Stress Level Predictor", layout="centered")
# st.title("🧠 Stress Level Prediction App")
# st.markdown("Enter your mental health and lifestyle information below:")

# # ------------------- Input Fields -------------------
# with st.form("stress_form"):
#     col1, col2 = st.columns(2)

#     with col1:
#         anxiety = st.slider("Anxiety Level", 0, 20, 10)
#         depression = st.slider("Depression", 0, 30, 10)
#         sleep_quality = st.slider("Sleep Quality", 0, 8, 4)
#         headache = st.slider("Headache Frequency", 0, 7, 3)

#     with col2:
#         academic_perf = st.slider("Academic Performance", 0, 5, 2)
#         career_concerns = st.slider("Future Career Concerns", 0, 5, 3)
#         social_support = st.slider("Social Support", 0, 5, 2)
#         living_conditions = st.slider("Living Conditions", 0, 5, 3)

#     submitted = st.form_submit_button("Predict Stress Level")

# # ------------------- Prediction Logic -------------------
# if submitted:
#     # Create input array
#     input_data = np.array([[anxiety, depression, sleep_quality, headache,
#                             academic_perf, career_concerns, social_support,
#                             living_conditions]])

#     # Optionally reduce influence of living_conditions
#     input_data[0][-1] = input_data[0][-1] * 0.5  # Scaled down to reduce impact

#     # Scale input
#     input_scaled = scaler.transform(input_data)

#     # Predict
#     prediction = model.predict(input_scaled)[0]
#     predicted_class = np.argmax(prediction)

#     stress_labels = ["Low Stress", "Moderate Stress", "High Stress"]
#     emojis = ["😌", "😐", "😫"]

#     st.subheader("🧾 Prediction Result")
#     st.success(f"**Predicted Stress Level:** {stress_labels[predicted_class]} {emojis[predicted_class]}")

#     # Display confidence scores
#     st.markdown("### 🔍 Prediction Confidence")
#     confidence_df = pd.DataFrame({
#         "Stress Level": stress_labels,
#         "Confidence": np.round(prediction * 100, 2)
#     })

#     st.bar_chart(confidence_df.set_index("Stress Level"))

#     st.markdown("✅ Tip: Try changing inputs to explore how they impact stress level.")

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib
# import matplotlib.pyplot as plt
# # # import streamlit as st

# # st.set_page_config(page_title="Cognifit", page_icon="🧠", layout="centered")

# # st.sidebar.success("👈 Choose a page to get started!")

# # st.title("Welcome to Cognifit 🧠")
# # st.markdown("""
# # This app predicts your **stress level** based on personal and lifestyle factors using a trained neural network.

# # 👈 Use the sidebar to:
# # - Predict your stress
# # """)
# # Load model
# try:
#     model = tf.keras.models.load_model("stress_model.h5")
#     scaler = joblib.load("scaler.pkl")
# except Exception as e:
#     st.error(f"Model or scaler not found: {e}")
#     st.stop()

# st.title("🧠 Cognifit: Stress Level Predictor")
# st.caption("This app predicts your mental health stress level based on lifestyle and health factors.")

# # Theme toggle (Dark mode)
# # theme = st.toggle("🌗 Dark Mode")
# # if theme:
# #     st.markdown("<style>body { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)

# # Layout
# col1, col2 = st.columns(2)

# with col1:
    
#     mental_health_history = st.radio("🧠 Past Mental Health Condition?", ["No", "Yes"])
#     anxiety = st.slider("😰 Anxiety Level", 0, 20, 10)
#     mental_health_history = 1 if mental_health_history == "Yes" else 0
#     depression = st.slider("😞 Depression Level", 0, 30, 10)
#     headache = st.slider("🤕 Headache Frequency", 0, 5,2)
#     breathing_problem = st.slider("😮‍💨 Breathing Difficulty", 0, 5, 2)

# with col2:
#     sleep_quality = st.slider("🛌 Sleep Quality", 0, 5, 2)
#     living_conditions = st.slider("🏠 Living Conditions", 0, 5, 2)
#     academic_performance = st.slider("📚 Academic Performance", 0, 5, 2)
#     study_load = st.slider("📖 Study Load", 0, 5, 2)
#     career_concerns = st.slider("💼 Career Concerns", 0, 5, 2)
#     extracurricular = st.slider("🎭 Extracurricular Activity", 0, 5, 2)

# input_data = np.array([[anxiety, mental_health_history, depression, headache, sleep_quality,
#                         breathing_problem, living_conditions, academic_performance,
#                         study_load, career_concerns, extracurricular]])

# if st.button("🎯 Predict Now"):
#     # st.info("Processing prediction...")
#     input_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_scaled)
#     predicted_class = int(np.argmax(prediction))

#     stress_labels = {0: "Low", 1: "Moderate", 2: "High"}
#     colors = {0: "🟢", 1: "🟠", 2: "🔴"}
#     emoji = colors[predicted_class]
#     level = stress_labels[predicted_class]

#     st.success(f"{emoji} Your Predicted Stress Level: *{level}*")

#     # Optional download
#     st.download_button("💾 Download Result", f"Stress Level: {level}", file_name="prediction.txt")