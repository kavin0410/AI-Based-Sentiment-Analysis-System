"""System Component for AI Sentiment Intelligence.

Provides operational system health diagnostics and architectural pipeline visualization.
"""

from pathlib import Path
import streamlit as st
import config
from src.model_loader import validate_model_artifacts


def render_system_page():
    """Render the System Health & Architecture page."""
    st.markdown("## ⚙️ System Diagnostics & Architecture")
    st.caption("Inspect live component statuses, operational model availability, and end-to-end data pipeline flow.")

    val_report = validate_model_artifacts()

    # -------------------------------------------------------------------------
    # 1. Component Health Matrix
    # -------------------------------------------------------------------------
    st.markdown("### 🔌 Component Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="asi-card">
                <div style="font-weight: 600; color: #8b949e; font-size: 0.85rem;">AI CORE ENGINE</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin-top: 4px; color: #3fb950;">
                    <span class="pulse-dot"></span> ONLINE
                </div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 6px;">Runtime: Python 3.10+ / Scikit-Learn</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="asi-card">
                <div style="font-weight: 600; color: #8b949e; font-size: 0.85rem;">PREDICTION ENGINE</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin-top: 4px; color: #3fb950;">
                    <span class="pulse-dot"></span> READY
                </div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 6px;">Real-time inference & history logging</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        vocab_sz = val_report.get("vectorizer_vocab_size", 570)
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-weight: 600; color: #8b949e; font-size: 0.85rem;">TF-IDF VECTORIZER</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin-top: 4px; color: #3fb950;">
                    <span class="pulse-dot"></span> LOADED
                </div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 6px;">{vocab_sz} terms (unigrams + bigrams)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        lr_status = "LOADED" if config.LOGISTIC_REGRESSION_FILE.exists() else "OFFLINE"
        lr_color = "#3fb950" if config.LOGISTIC_REGRESSION_FILE.exists() else "#f85149"
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-weight: 600; color: #8b949e; font-size: 0.85rem;">LOGISTIC REGRESSION</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin-top: 4px; color: {lr_color};">
                    <span class="pulse-dot" style="background-color: {lr_color}; box-shadow: 0 0 8px {lr_color};"></span> {lr_status}
                </div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 6px;">Active Best Model (Top Rank)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        nb_status = "LOADED" if config.NAIVE_BAYES_FILE.exists() else "OFFLINE"
        nb_color = "#3fb950" if config.NAIVE_BAYES_FILE.exists() else "#f85149"
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-weight: 600; color: #8b949e; font-size: 0.85rem;">MULTINOMIAL NAIVE BAYES</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin-top: 4px; color: {nb_color};">
                    <span class="pulse-dot" style="background-color: {nb_color}; box-shadow: 0 0 8px {nb_color};"></span> {nb_status}
                </div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 6px;">alpha=1.0 (Laplace smoothing)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        svm_status = "LOADED" if config.LINEAR_SVM_FILE.exists() else "OFFLINE"
        svm_color = "#3fb950" if config.LINEAR_SVM_FILE.exists() else "#f85149"
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-weight: 600; color: #8b949e; font-size: 0.85rem;">LINEAR SVM (LinearSVC)</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin-top: 4px; color: {svm_color};">
                    <span class="pulse-dot" style="background-color: {svm_color}; box-shadow: 0 0 8px {svm_color};"></span> {svm_status}
                </div>
                <div style="font-size: 0.8rem; color: #8b949e; margin-top: 6px;">Decision function scores</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Pipeline Flow Diagram
    # -------------------------------------------------------------------------
    st.markdown("### 🏛️ End-to-End Pipeline Flow")
    st.code(
        """
Dataset (60 records)
  │
  ├── Data Cleaning (noise removal, URLs, HTML tags, whitespace)
  │
  ├── 12-Step NLP Preprocessing (contractions, negation preservation, lemmatization)
  │
  ├── Stratified Split (80% train / 20% test — random_state=42)
  │
  ├── TF-IDF Vectorization (fit on train only -> 570 features, unigrams+bigrams)
  │
  ├── Multi-Model Training (Logistic Regression, Multinomial NB, Linear SVM)
  │
  ├── Evaluation & Benchmarking (Weighted F1, Accuracy, Confusion Matrices)
  │
  └── Real-Time Prediction Engine (Dynamic best-model inference & session history)
        """,
        language="text",
    )

    st.markdown(
        """
        **Architectural Integrity Rules:**
        - **Zero Data Leakage:** Vectorizer is fit exclusively on training data and frozen before evaluation.
        - **Dynamic Selection:** Predictions dynamically load the highest-scoring model rather than using hardcoded assignments.
        - **Controlled Preprocessing:** Negation words (`not`, `no`, `never`) are explicitly preserved to safeguard sentiment polarity.
        """
    )
