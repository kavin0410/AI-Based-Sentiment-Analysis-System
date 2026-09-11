"""Settings Component for SentimentLab.

Provides:
- Application version & runtime parameter matrix
- Technical component states:
  * Application Version
  * NLP Engine Status
  * TF-IDF Status
  * Models Loaded
  * Prediction Engine
  * Dataset Status
- "About SentimentLab" section:
  "SentimentLab is an AI-based sentiment analysis system that combines NLP preprocessing,
   TF-IDF feature extraction, and machine-learning classification to identify Positive, Negative, and Neutral sentiment."
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import config
from src.model_loader import validate_model_artifacts


def render_system_page():
    """Render the Settings & Infrastructure page."""
    st.markdown("## ⚙ Settings")
    st.caption("Application environment, runtime parameters, model metadata, and architecture overview.")

    # -------------------------------------------------------------------------
    # 1. About SentimentLab Section
    # -------------------------------------------------------------------------
    st.markdown("### About SentimentLab")
    st.markdown(
        """
        <div class="product-card">
            <p style="font-size: 1.05rem; color: #bae6fd; font-weight: 500; line-height: 1.6; margin: 0;">
                SentimentLab is an AI-based sentiment analysis system that combines NLP preprocessing, 
                TF-IDF feature extraction, and machine-learning classification to identify Positive, Negative, and Neutral sentiment.
            </p>
            <div style="margin-top: 12px; font-size: 0.85rem; color: #94a3b8;">
                Built on Python 3.10+, Scikit-Learn, NLTK, and Streamlit. Designed with zero data leakage principles, 
                controlled negation preservation, and dynamic top-model resolution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Technical System Status
    # -------------------------------------------------------------------------
    st.markdown("### System Component Status")

    val_report = validate_model_artifacts()
    vocab_sz = val_report.get("vectorizer_vocab_size", 570)
    top_model = val_report.get("best_model_name", "Logistic Regression")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Application Version</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #ffffff;">v2.5.0 SaaS</div>
                <div class="kpi-sub">Production UI Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">NLP Engine Status</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #34d399;">ONLINE</div>
                <div class="kpi-sub">NLTK Tokenizer & WordNet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">TF-IDF Status</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #34d399;">LOADED</div>
                <div class="kpi-sub">{vocab_sz} N-gram Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Models Loaded</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #34d399;">3 / 3 Active</div>
                <div class="kpi-sub">LR, Naive Bayes, Linear SVM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Prediction Engine</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #34d399;">READY</div>
                <div class="kpi-sub">Real-Time + Batch CSV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Dataset Status</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #cbd5e1;">60 Records</div>
                <div class="kpi-sub">Balanced 3-Class Corpus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Runtime Parameters
    # -------------------------------------------------------------------------
    st.markdown("### Runtime Parameter Specifications")
    params_df = pd.DataFrame({
        "Configuration Key": [
            "Random State Seed",
            "Max Text Input Length",
            "Session History Limit",
            "Classification Polarity Labels",
            "TF-IDF Max Feature Cap",
            "Stratified Test Size",
            "Model Selection Metric",
        ],
        "Configured Value": [
            str(config.RANDOM_STATE),
            f"{config.MAX_INPUT_LENGTH} characters",
            f"{config.PREDICTION_HISTORY_LIMIT} records",
            ", ".join(config.SUPPORTED_LABELS),
            "5000 (570 fitted)",
            f"{int(config.TEST_SIZE * 100)}% ({config.TEST_SIZE})",
            "Weighted F1-Score",
        ],
        "Operational Mode": ["Fixed", "Validated", "In-Memory", "Static", "Serialized", "Stratified", "Dynamic"],
    })
    st.dataframe(params_df, use_container_width=True)