import streamlit as st
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("⚠️ Model or Vectorizer files not found. Please upload the correct files.")
    st.stop()

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Header and Instructions
st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>📰 Fake News Detection Tool</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>Check if a news article is Real or Fake instantly!</p>",
            unsafe_allow_html=True)

# Text Input Section
st.markdown("### 📝 Paste or type your news article below:")
text_input = st.text_area("", height=200, placeholder="Enter news content here...")

# File Upload Feature
uploaded_file = st.file_uploader("Or Upload a .txt News File", type=["txt"])
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.text_area("News Content from File", content, height=200)

# Detect Button with Loading Spinner
if st.button("🔍 Detect News"):
    if text_input.strip() == "" and not uploaded_file:
        st.warning("⚠️ Please enter or upload some text.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)  # Fake delay for the spinner effect
            input_vector = vectorizer.transform([text_input if text_input.strip() != "" else content])
            prediction = model.predict(input_vector)[0]
            probability = model.predict_proba(input_vector)[0]

            # Store history
            st.session_state.history.append({
                "text": text_input if text_input.strip() != "" else content,
                "prediction": prediction,
                "probability": max(probability) * 100
            })

            # Show last 5 predictions
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[-5:]

            # Show Prediction History
            st.markdown("### 🕒 Prediction History")
            for i, item in enumerate(st.session_state.history):
                st.write(f"{i + 1}. **{item['prediction']}** - {item['probability']:.2f}% Probability")

            # Display Prediction and Probability
            st.write(f"Prediction: **{prediction}**")
            st.write(f"Probability: **{max(probability) * 100:.2f}%**")

            if prediction.upper() == "FAKE":
                st.error("❌ This news is **FAKE**.")
            else:
                st.success("✅ This news is **REAL**.")
