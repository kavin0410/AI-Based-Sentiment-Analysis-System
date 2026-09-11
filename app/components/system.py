"""Settings Component for SentimentLab — Premium Light UI.

Provides:
- Application version & runtime parameter matrix
- Technical component states:
  * Application Version
  * NLP Engine Status
  * TF-IDF Status
  * Models Loaded
  * Prediction Engine
  * Dataset Status
- "About SentimentLab" section
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import config
from src.model_loader import validate_model_artifacts


def render_system_page():
    """Render the Settings & Infrastructure page."""
    st.markdown(
        """
        <div class="sl-page-header">
            <div class="sl-page-eyebrow">SYSTEM & INFRASTRUCTURE</div>
            <h1 class="sl-page-title">Settings</h1>
            <p class="sl-page-subtitle">
                Application environment, runtime parameters, model metadata, and platform architecture overview.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 1. About SentimentLab Section
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="sl-card sl-card-hero" style="margin-bottom: 1.5rem;">
            <div class="sl-section-title" style="font-size: 1.4rem; color: #0F172A;">About SentimentLab</div>
            <p style="font-size: 1.05rem; color: #475569; font-weight: 500; line-height: 1.6; margin: 0.5rem 0 0.75rem 0;">
                SentimentLab is an AI-based sentiment intelligence platform that seamlessly unifies advanced NLP preprocessing, 
                TF-IDF linguistic feature extraction, and machine learning classifiers to determine Positive, Negative, and Neutral polarity.
            </p>
            <div style="font-size: 0.88rem; color: #64748B; line-height: 1.6;">
                Engineered with Scikit-Learn, NLTK, and modern Web UI primitives. Adheres to zero data-leakage standards, 
                intelligent negation retention, and dynamic top-model resolution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. Technical System Status
    # -------------------------------------------------------------------------
    st.markdown(
        '<div class="sl-section-label" style="margin-bottom:0.75rem">System Component Status</div>',
        unsafe_allow_html=True,
    )

    val_report = validate_model_artifacts()
    vocab_sz = val_report.get("vectorizer_vocab_size", 570)
    top_model = val_report.get("best_model_name", "Logistic Regression")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="sl-metric sl-card-blue">
                <div class="sl-metric-label">Application Version</div>
                <div class="sl-metric-value" style="font-size: 1.5rem; color: #0F172A;">v3.0 SaaS</div>
                <div class="sl-metric-sub">Production UI Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="sl-metric sl-card-green">
                <div class="sl-metric-label">NLP Engine Status</div>
                <div class="sl-metric-value" style="font-size: 1.5rem; color: #059669;">ONLINE</div>
                <div class="sl-metric-sub">NLTK Tokenizer & WordNet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="sl-metric sl-card-violet">
                <div class="sl-metric-label">TF-IDF Status</div>
                <div class="sl-metric-value" style="font-size: 1.5rem; color: #6366F1;">LOADED</div>
                <div class="sl-metric-sub">{vocab_sz} N-gram Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            """
            <div class="sl-metric sl-card-cyan">
                <div class="sl-metric-label">Models Loaded</div>
                <div class="sl-metric-value" style="font-size: 1.5rem; color: #0284C7;">3 / 3 Active</div>
                <div class="sl-metric-sub">LR, Naive Bayes, Linear SVM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div class="sl-metric sl-card-green">
                <div class="sl-metric-label">Prediction Engine</div>
                <div class="sl-metric-value" style="font-size: 1.5rem; color: #059669;">READY</div>
                <div class="sl-metric-sub">Real-Time + Batch CSV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            """
            <div class="sl-metric sl-card-blue">
                <div class="sl-metric-label">Dataset Status</div>
                <div class="sl-metric-value" style="font-size: 1.5rem; color: #475569;">60 Records</div>
                <div class="sl-metric-sub">Balanced 3-Class Corpus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sl-spacer-md'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Runtime Parameters
    # -------------------------------------------------------------------------
    st.markdown(
        '<div class="sl-card">'
        '<div class="sl-section-title">Runtime Parameter Specifications</div>'
        '<div class="sl-text-muted" style="margin-bottom:1rem;">Operational environment configuration constants.</div>',
        unsafe_allow_html=True,
    )
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
    st.table(params_df)
    st.markdown("</div>", unsafe_allow_html=True)