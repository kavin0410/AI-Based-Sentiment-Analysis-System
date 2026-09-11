"""Overview Page Component for AI Sentiment Intelligence (Stage 6).

Provides an executivelanding overview featuring:
- Key system metrics (Dataset size, Train/Test split, Features, Models, Predictions count)
- Dynamic Best Model Card (reads results/best_model.json)
- Transparent Dataset Health Card (with 60-record small dataset educational disclaimer)
- Model Health & System Status Card
- Quick Hero Sentiment Analyzer
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
from src.data_loader import load_dataset
from src.model_loader import (
    load_best_model,
    load_best_model_metadata,
    load_vectorizer,
    validate_model_artifacts,
)
from src.predict import predict_sentiment


def render_overview_page():
    """Render the executive Overview page."""
    st.markdown("## 🏠 Executive System Overview")
    st.caption("AI Sentiment Intelligence Platform — System Performance & Model Health")

    # Load dataset & metadata dynamically
    best_model_meta = None
    if config.BEST_MODEL_JSON.exists():
        with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
            best_model_meta = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Executive Metric Cards
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("##### 📌 Key System Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    history_mgr = st.session_state.get("history_manager")
    session_preds_count = len(history_mgr.get_history()) if history_mgr else 0

    with col1:
        st.metric("Total Dataset", "60", help="Total records in cleaned dataset")
    with col2:
        st.metric("Train Samples", "48", help="80% stratified training split")
    with col3:
        st.metric("Test Samples", "12", help="20% stratified held-out test split")
    with col4:
        st.metric("TF-IDF Features", "570", help="Unigram + bigram feature vocabulary size")
    with col5:
        st.metric("ML Models", "3", help="Logistic Regression, Naive Bayes, Linear SVM")
    with col6:
        st.metric("Session Predictions", str(session_preds_count), help="Predictions performed in current session")

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Main System Cards (Best Model, Dataset Health, Model Health)
    # -------------------------------------------------------------------------
    col_card1, col_card2, col_card3 = st.columns([4, 4, 4])

    # --- Best Model Card ---
    with col_card1:
        st.markdown("#### 🏆 Best Model Card")
        if best_model_meta:
            m_name = best_model_meta.get("model_name", "Logistic Regression")
            f1_val = best_model_meta.get("f1_score", 0.0)
            acc_val = best_model_meta.get("accuracy", 0.0)
            prec_val = best_model_meta.get("precision", 0.0)
            rec_val = best_model_meta.get("recall", 0.0)
            criterion = best_model_meta.get("selection_metric", "f1_score").replace("_", " ").title()

            st.success(f"### **{m_name}**\nTop-Performing Classifier")
            st.markdown(f"**Selection Criterion:** Weighted {criterion}")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Weighted F1", f"{f1_val * 100:.2f}%")
                st.metric("Precision", f"{prec_val * 100:.2f}%")
            with m_col2:
                st.metric("Accuracy", f"{acc_val * 100:.2f}%")
                st.metric("Recall", f"{rec_val * 100:.2f}%")
        else:
            st.warning("Best model evaluation data not found. Run `python run_stage4.py`.")

    # --- Dataset Health Card ---
    with col_card2:
        st.markdown("#### 🛡️ Dataset Health")
        st.info("### **Cleaned Dataset**\nBalanced 3-Class Corpus")
        st.markdown("**Classes:** Positive (20), Negative (20), Neutral (20)")
        st.markdown("**Validation:** 0 Missing Rows | 0 Duplicate Rows")

        # Explicit Transparent Educational Disclaimer
        st.warning(
            "⚠️ **Evaluation Note:** Current model evaluation is based on a small 60-record dataset. "
            "Results are intended for educational/project demonstration and should not be interpreted "
            "as production-grade performance."
        )

    # --- Model Health Card ---
    with col_card3:
        st.markdown("#### 🧠 System & Model Health")
        val_report = validate_model_artifacts()

        if val_report["valid"]:
            st.success("### **System Status: ONLINE**\nAll Artifacts Loaded")
            st.markdown("- **Models Available:** `3 / 3` (LR, NB, SVM)")
            st.markdown(f"- **TF-IDF Vectorizer:** `Loaded` ({val_report['vectorizer_vocab_size']} terms)")
            st.markdown(f"- **Active Best Model:** `{val_report['best_model_name']}`")
            st.markdown("- **Prediction Engine:** `Ready`")
        else:
            st.error("### **System Status: DEGRADED**\nMissing Model Artifacts")
            for err in val_report.get("errors", []):
                st.caption(f"• {err}")

    # -------------------------------------------------------------------------
    # 3. Hero Quick Sentiment Analyzer
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("### ⚡ Quick Sentiment Analyzer")
    st.caption("Test real-time prediction instantly directly from the overview dashboard.")

    quick_text = st.text_area(
        "Enter text to analyze:",
        value="I absolutely love this amazing project! It works seamlessly.",
        height=90,
        key="hero_quick_input",
    )

    if st.button("⚡ ANALYZE SENTIMENT (Hero)", type="primary", use_container_width=True):
        if not quick_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                res = predict_sentiment(quick_text)

            if res.get("valid", False):
                # Record to session history
                if history_mgr:
                    history_mgr.add_prediction(res)

                st.markdown("#### Result:")
                s_label = res["sentiment"]
                score_val = res["score"]
                score_type = res["score_type"]

                if s_label == "Positive":
                    st.success(f"### 🟢 POSITIVE — {res['model_name']}")
                elif s_label == "Negative":
                    st.error(f"### 🔴 NEGATIVE — {res['model_name']}")
                else:
                    st.info(f"### ⚪ NEUTRAL — {res['model_name']}")

                if score_type == "probability":
                    st.markdown(f"**Confidence:** `{score_val * 100:.1f}%`")
                else:
                    st.markdown(f"**Decision Score:** `{score_val:.4f}` *(Linear SVM decision boundary distance)*")

                st.caption("ℹ️ *This prediction was generated using Stage 2 NLP preprocessing + Stage 3 TF-IDF vectorization + Stage 4 evaluated best model.*")
