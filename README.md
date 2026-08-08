# Explainable-AI-Based-Medical-Report-Diagnosis-Prediction-Using-NLP
Explainable AI-Based Medical Report Diagnosis Prediction uses NLP and machine learning to analyze medical reports and predict possible diagnoses. TF-IDF extracts important text features, while LIME and SHAP provide model explanations, improving transparency and interpretability. The system can be deployed using Streamlit.

# 📌 Project Overview

Explainable AI-Based Medical Report Diagnosis Prediction Using NLP is an end-to-end Artificial Intelligence project that analyzes unstructured medical report text and predicts a possible diagnosis/class using Natural Language Processing (NLP) and Machine Learning.

Unlike a traditional black-box prediction system, this project integrates Explainable AI (XAI) techniques to provide insights into why a particular prediction was generated.

The project converts medical text into numerical representations using TF-IDF, applies a trained machine-learning classifier, generates prediction probabilities, and explores model explanations using LIME and SHAP.

⚠️ Important Medical DisclaimerThis project is developed for educational, academic, and research purposes only. It is not a medical device, does not provide professional medical advice, and must not be used as a substitute for diagnosis or treatment by a qualified healthcare professional.

## ✨ Key Features


📝 Medical Text Input

  Accepts medical reports written in natural language

🧹 NLP Processing

  Processes and transforms textual medical information

🔤 TF-IDF

   Converts text into numerical feature vectors

🤖 ML Prediction

   Predicts the learned diagnosis/class

📊 Probability

   Displays prediction probability when supported by the model

🔎 LIME

   Explains individual predictions using important words/features

🧠 SHAP

   Provides feature-level model explainability

💾 Model Saving
   Stores trained model and TF-IDF vectorizer using Joblib

🌐 Streamlit Ready
   Can be deployed as an interactive web application
