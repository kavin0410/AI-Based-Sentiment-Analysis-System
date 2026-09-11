"""Overview Component for AI Sentiment Intelligence.

Provides an executive AI SaaS landing overview:
- Header & subtitle
- System pulse indicator
- 4 Key metric cards
- Main workspace: Quick Sentiment Analyzer (Left) + Live Result Card (Right)
"""

import json
from pathlib import Path
import streamlit as st
import config
from src.model_loader import (
    load_best_model,
    load_vectorizer,
    validate_model_artifacts,
)
from src.predict import predict_sentiment


def render_overview_page():
    """Render the polished executive Overview page."""
    # Subtitle / Tagline
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <p style="font-size: 1.15rem; color: #8b949e; margin: 0; font-style: italic;">
                "Understand what people feel through machine learning."
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 1. Top Executive Metric Cards
    # -------------------------------------------------------------------------
    best_meta = {}
    if config.BEST_MODEL_JSON.exists():
        try:
            with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
        except Exception:
            pass

    acc_val = best_meta.get("accuracy", 0.3333)
    best_model_name = best_meta.get("model_name", "Logistic Regression")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", "60 Records", help="Educational corpus (48 train / 12 test)")
    with col2:
        st.metric("TF-IDF Features", "570 Terms", help="Vocabulary extracted with unigrams + bigrams")
    with col3:
        st.metric("Models Available", "3 Classifiers", help="Logistic Regression, Naive Bayes, Linear SVM")
    with col4:
        st.metric("Best Model Accuracy", f"{acc_val * 100:.2f}%", f"{best_model_name} (F1: {best_meta.get('f1_score', 0.3276)*100:.2f}%)")

    st.markdown(
        """
        <div style="background: rgba(56, 139, 253, 0.08); border-left: 3px solid #388bfd; padding: 8px 12px; border-radius: 4px; font-size: 0.85rem; color: #8b949e; margin: 1rem 0;">
            ℹ️ <strong>Evaluation Note:</strong> All metrics shown reflect the real held-out test split of the current 60-record educational dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Main Workspace: Quick Analyzer (Left) + Live Result Card (Right)
    # -------------------------------------------------------------------------
    st.markdown("### ⚡ Quick Sentiment Analyzer")
    col_left, col_right = st.columns([5, 5])

    with col_left:
        quick_text = st.text_area(
            "Enter text to test real-time sentiment:",
            value=st.session_state.get("overview_input_text", "The customer support team resolved my issue quickly and was very helpful!"),
            height=130,
            key="overview_quick_textarea",
            placeholder="Type or paste any text to analyze...",
        )

        col_btn1, col_btn2 = st.columns([3, 2])
        with col_btn1:
            btn_run = st.button("✨ Analyze Now", type="primary", use_container_width=True, key="btn_overview_analyze")
        with col_btn2:
            if st.button("Clear", use_container_width=True, key="btn_overview_clear"):
                st.session_state["overview_input_text"] = ""
                st.rerun()

    with col_right:
        st.markdown("##### Live Result")
        result = None

        if btn_run and quick_text.strip():
            with st.spinner("Classifying sentiment..."):
                result = predict_sentiment(quick_text.strip())
                st.session_state["last_overview_result"] = result
                # Log to session history manager
                history_mgr = st.session_state.get("history_manager")
                if history_mgr and result.get("valid"):
                    history_mgr.add_prediction(result)
        elif "last_overview_result" in st.session_state:
            result = st.session_state["last_overview_result"]

        if result and result.get("valid"):
            sentiment = result["sentiment"]
            score = result["score"]
            score_type = result["score_type"]
            m_used = result.get("model_name", best_model_name)
            probs = result.get("probabilities", {})

            badge_class = "badge-neutral"
            if sentiment == "Positive":
                badge_class = "badge-positive"
            elif sentiment == "Negative":
                badge_class = "badge-negative"

            st.markdown(
                f"""
                <div class="asi-card" style="margin-top: 0.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span class="{badge_class}" style="font-size: 1.1rem; padding: 6px 16px;">
                            {sentiment.upper()}
                        </span>
                        <span style="font-size: 0.85rem; color: #8b949e;">Model: <strong>{m_used}</strong></span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if score_type == "probability":
                st.metric("Confidence Score", f"{score * 100:.1f}%")
                st.caption("Class Probability Distribution:")
                for cls in ["Positive", "Neutral", "Negative"]:
                    p_val = probs.get(cls, 0.0)
                    st.write(f"**{cls}:** {p_val * 100:.1f}%")
                    st.progress(min(max(p_val, 0.0), 1.0))
            else:
                st.metric("Decision Score", f"{score:.4f}")
                st.caption("Raw decision score (Linear SVM) — not a probability.")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="asi-card" style="text-align: center; padding: 2.5rem 1rem; color: #8b949e;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">✨</div>
                    <div style="font-size: 1rem; font-weight: 500;">Ready to analyze</div>
                    <div style="font-size: 0.85rem; margin-top: 0.25rem;">Enter text on the left and click "Analyze Now"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
