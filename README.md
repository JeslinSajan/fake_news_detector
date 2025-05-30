# 📰 Fake News Detection Tool

A web application that detects whether a news article is **FAKE** or **REAL**, using a trained machine learning model and Streamlit for the frontend.

---

## 🚀 Features

- Paste or upload a news article for analysis
- Get instant prediction (FAKE / REAL) with confidence score
- Beautiful UI with custom styling
- Built with Streamlit and trained using PassiveAggressiveClassifier

---

## 🧠 Machine Learning Model

- Dataset: `Fake.csv` and `True.csv`
- Model: `PassiveAggressiveClassifier`
- Training script: `train_model.py`
- Saved model: `fake_news_model.pkl`, `vectorizer.pkl`

---

## 🛠 How to Run Locally

### 📦 Step 1: Install Requirements

```bash
pip install -r requirements.txt

