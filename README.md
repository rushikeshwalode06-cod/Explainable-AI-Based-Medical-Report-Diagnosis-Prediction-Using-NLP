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

## 🎯 Project Objectives

1. The main objectives of this project are:
2. Analyze unstructured medical reports using NLP.
3. Extract meaningful textual features from medical information.
4. Convert medical text into numerical vectors using TF-IDF.
5. Train a machine-learning classification model.
6. Predict the diagnosis/class from unseen medical reports.
7. Display prediction probability where supported.
8. Apply Explainable AI to understand model behavior.
9. Save trained components for reuse and deployment.
10. Provide a foundation for an interactive AI-based medical text application.

## 🧩 System Architecture

                    ┌───────────────────────┐
                    │   Medical Report      │
                    │      Text Input       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Text Preprocessing  │
                    │ Cleaning / Processing │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    TF-IDF Vectorizer  │
                    │ Feature Extraction    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Machine Learning      │
                    │ Classification Model  │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌─────────────────┐     ┌──────────────────┐
          │ Prediction      │     │ Prediction       │
          │ Diagnosis/Class  │     │ Probability      │
          └────────┬────────┘     └────────┬─────────┘
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    ┌───────────────────────┐
                    │ Explainable AI (XAI) │
                    │ LIME + SHAP         
                    └───────────────────────┘
  ## 🔬 Methodology

1. Data Collection

Medical report text and corresponding diagnosis/class labels are used as the basis for supervised machine-learning classification.

2. Text Preprocessing

The textual reports are prepared for machine learning by applying suitable NLP preprocessing operations.
Typical preprocessing may include:

Lowercasing

Removing unnecessary characters

Removing unwanted spaces

Tokenization

Stop-word handling

Text normalization

3. Feature Extraction — TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) transforms medical text into numerical feature vectors.

It gives higher importance to terms that are useful for distinguishing documents while reducing the importance of very common terms.

4. Model Training

The transformed text features are supplied to a machine-learning classification model.

The trained classifier learns relationships between medical text features and the target diagnosis/class.

5. Prediction

For a new medical report:

Medical Report
      ↓
TF-IDF Transformation
      ↓
Trained Model
      ↓
Predicted Class

6. Explainability

The project uses XAI techniques to make predictions easier to understand.

LIME focuses on explaining individual predictions.

SHAP analyzes feature contributions to model behavior.
