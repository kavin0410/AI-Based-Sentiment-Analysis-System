"""Architecture Component for AI Sentiment Intelligence (Stage 6).

Provides an interactive breakdown of the end-to-end system architecture and data flow.
"""

import streamlit as st


def render_architecture_page():
    """Render the System Architecture page."""
    st.markdown("## 🏛️ System Architecture & Data Flow")
    st.caption("Complete end-to-end machine learning pipeline architecture from raw data ingestion to real-time Streamlit web deployment.")

    st.code(
        """
┌─────────────────────────┐
│     Raw Dataset CSV     │  (data/raw/sample_data.csv - 60 records)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Data Validation      │  (src/data_loader.py: schema, missing, label checks)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     Data Cleaning       │  (src/data_cleaning.py: lowercasing, URLs, HTML, emails)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   NLP Preprocessing     │  (src/preprocessing.py: contractions, controlled stops, lemmatization)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Cleaned Dataset CSV   │  (data/processed/cleaned_dataset.csv)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Stratified 80/20 Split│  (data_loader.py: random_state=42, 48 train / 12 test)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   TF-IDF Vectorizer     │  (src/feature_extraction.py: 570 features, unigrams+bigrams, sublinear_tf)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Classifier Models     │  (src/train_model.py: Logistic Regression, Naive Bayes, Linear SVM)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Model Evaluation      │  (src/evaluate_model.py: Metrics, Confusion Matrices, Error Analysis)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Dynamic Best Model      │  (src/model_loader.py: reads results/best_model.json)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Real-Time Predictor     │  (src/predict.py: Live Inference, Confidence Scores, Session History)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Streamlit Web Dashboard │  (app/app.py & app/components/: 8 Logical Platform Views)
└─────────────────────────┘
        """,
        language="text",
    )

    st.divider()
    st.markdown("### 🛠️ Architectural Principles")
    st.markdown(
        """
        - **Zero Data Leakage:** Train/Test split occurs strictly prior to TF-IDF vectorizer fitting.
        - **Dynamic Model Resolution:** Inference engine resolves the top model from Stage 4 evaluation (`results/best_model.json`) rather than hardcoding model names.
        - **Strict Confidence Distinction:** Probability models (`predict_proba`) display confidence percentages, while non-probability models (`LinearSVC`) display raw decision scores with explicit disclaimers.
        - **Stateful In-Memory History:** Prediction history is managed in session memory with configurable size limits (`PREDICTION_HISTORY_LIMIT = 50`) and instant CSV export.
        """
    )
