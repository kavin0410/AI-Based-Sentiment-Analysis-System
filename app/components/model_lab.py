"""Model Lab Component for AI Sentiment Intelligence (Stage 6).

Provides technical model details, hyperparameter specifications, and dynamic selection mechanism explanations:
- Logistic Regression, Multinomial Naive Bayes, Linear SVM
- Best model selection logic (Primary: Weighted F1, Secondary: Accuracy)
"""

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config


def render_model_lab_page():
    """Render the Model Lab page."""
    st.markdown("## 🧠 Model Lab & Selection Intelligence")
    st.caption("Machine learning model architecture, hyperparameters, persistence status, and automated selection logic.")

    # 1. Best Model Selection Explanation
    st.divider()
    st.markdown("### ⚙️ How the Best Model is Selected")
    st.markdown(
        """
        The system employs an **automated, objective best-model selection pipeline** (Stage 4) based on empirical test set evaluation:
        1. **Primary Metric:** **Weighted F1-Score** — Measures the harmonic mean of Precision and Recall across all 3 classes, accounting for support.
        2. **Secondary Metric (Tie-Breaker):** **Overall Accuracy** — Used if two models yield identical weighted F1-scores.
        3. **No Hardcoding:** The selected model name is stored in `results/best_model.json` and loaded dynamically by the inference engine (`src/model_loader.py`).
        """
    )

    # 2. Model Registry Overview Cards
    st.divider()
    st.markdown("### 🤖 Trained Model Registry")

    best_meta = None
    if config.BEST_MODEL_JSON.exists():
        with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
            best_meta = json.load(f)

    best_name = best_meta.get("model_name") if best_meta else "Logistic Regression"

    col_m1, col_m2, col_m3 = st.columns(3)

    # Logistic Regression
    with col_m1:
        badge = "🏆 BEST MODEL" if best_name == "Logistic Regression" else "TRAINED"
        st.info(f"### Logistic Regression\nStatus: `{badge}`")
        st.markdown("**Algorithm:** `sklearn.linear_model.LogisticRegression`")
        st.markdown("**Hyperparameters:** `solver='lbfgs'`, `max_iter=1000`, `C=1.0`")
        st.markdown("**Artifact:** `models/logistic_regression.pkl` (14.7 KB)")
        st.markdown("**Probability Support:** `Yes` (`predict_proba`)")

    # Multinomial Naive Bayes
    with col_m2:
        badge = "🏆 BEST MODEL" if best_name == "Multinomial Naive Bayes" else "TRAINED"
        st.info(f"### Multinomial Naive Bayes\nStatus: `{badge}`")
        st.markdown("**Algorithm:** `sklearn.naive_bayes.MultinomialNB`")
        st.markdown("**Hyperparameters:** `alpha=1.0` (Laplace Smoothing)")
        st.markdown("**Artifact:** `models/naive_bayes.pkl` (28.2 KB)")
        st.markdown("**Probability Support:** `Yes` (`predict_proba`)")

    # Linear SVM
    with col_m3:
        badge = "🏆 BEST MODEL" if best_name == "Linear SVM" else "TRAINED"
        st.info(f"### Linear SVM\nStatus: `{badge}`")
        st.markdown("**Algorithm:** `sklearn.svm.LinearSVC`")
        st.markdown("**Hyperparameters:** `C=1.0`, `dual='auto'`, `max_iter=2000`")
        st.markdown("**Artifact:** `models/linear_svm.pkl` (14.5 KB)")
        st.markdown("**Probability Support:** `No` (Uses `decision_function`)")

    # 3. Model Metadata Inspection
    st.divider()
    st.markdown("### 📋 Model Training Metadata (`models/model_metadata.json`)")
    if config.MODEL_METADATA_FILE.exists():
        with open(config.MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        st.json(meta)
    else:
        st.warning("Model metadata file not found.")
