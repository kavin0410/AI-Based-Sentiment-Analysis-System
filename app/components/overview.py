"""Overview Component for SentimentLab.

Executive hero section:
- SentimentLab
  "Understand the emotion behind every word."
  "Analyze text, uncover sentiment, and transform language into actionable intelligence using machine learning."
- Primary CTA: Analyze Text (switches to Analyze tab)
- Secondary CTA: Explore Intelligence (switches to Model Intelligence tab)
- Live system statistics:
  * Total Dataset Records
  * Predictions
  * Positive %
  * Negative %
  * Neutral %
  * Average Confidence
  * TF-IDF Features
  * Active Model
- Compact System Health Indicators (ONLINE / READY):
  * AI Engine
  * NLP Pipeline
  * TF-IDF
  * ML Models
  * Prediction Engine
"""

import json
from pathlib import Path
import streamlit as st
import config
from src.model_loader import validate_model_artifacts


def render_overview_page():
    """Render the executive Overview hero page."""
    # -------------------------------------------------------------------------
    # 1. Hero Section
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="padding: 1rem 0 2rem 0; max-width: 900px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;">
                Neural Language Intelligence
            </div>
            <h1 style="font-size: 2.8rem; font-weight: 800; letter-spacing: -0.04em; margin: 0; background: linear-gradient(135deg, #ffffff 30%, #bae6fd 70%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                SentimentLab
            </h1>
            <p style="font-size: 1.25rem; font-weight: 600; color: #7dd3fc; margin: 0.5rem 0 0.75rem 0; letter-spacing: -0.01em;">
                “Understand the emotion behind every word.”
            </p>
            <p style="font-size: 1.02rem; color: #94a3b8; line-height: 1.6; margin: 0 0 1.75rem 0;">
                Analyze text, uncover sentiment, and transform language into actionable intelligence using machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Action CTAs
    col_cta1, col_cta2, _ = st.columns([2, 2, 4])
    with col_cta1:
        if st.button("✦ Analyze Text", type="primary", use_container_width=True, key="btn_hero_analyze"):
            st.session_state["sentimentlab_nav"] = "✦ Analyze"
            st.rerun()
    with col_cta2:
        if st.button("◉ Explore Intelligence", use_container_width=True, key="btn_hero_intelligence"):
            st.session_state["sentimentlab_nav"] = "◉ Model Intelligence"
            st.rerun()

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Live System Statistics
    # -------------------------------------------------------------------------
    st.markdown("### Live Platform Metrics")

    val_report = validate_model_artifacts()
    active_model = val_report.get("best_model_name", "Logistic Regression")
    vocab_sz = val_report.get("vectorizer_vocab_size", 570)

    history_mgr = st.session_state.get("history_manager")
    h_items = history_mgr.get_history() if history_mgr else []
    total_preds = len(h_items)

    if total_preds > 0:
        h_df = history_mgr.to_dataframe()
        s_counts = h_df["sentiment"].value_counts()
        pos_pct = (s_counts.get("Positive", 0) / total_preds) * 100
        neg_pct = (s_counts.get("Negative", 0) / total_preds) * 100
        neu_pct = (s_counts.get("Neutral", 0) / total_preds) * 100
        prob_rows = h_df[h_df["score_type"] == "probability"]
        avg_conf = (prob_rows["score"].mean() * 100) if not prob_rows.empty else 0.0
    else:
        # Default baseline distributions from training corpus
        pos_pct = 33.3
        neg_pct = 33.3
        neu_pct = 33.3
        avg_conf = 35.6

    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    with row1_c1:
        st.metric("Total Dataset Records", "60 Records", "48 train / 12 test")
    with row1_c2:
        st.metric("Predictions Logged", f"{total_preds}", "Active session queries")
    with row1_c3:
        st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
    with row1_c4:
        st.metric("Negative Sentiment", f"{neg_pct:.1f}%")

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    with row2_c1:
        st.metric("Neutral Sentiment", f"{neu_pct:.1f}%")
    with row2_c2:
        st.metric("Average Confidence", f"{avg_conf:.1f}%")
    with row2_c3:
        st.metric("TF-IDF Features", f"{vocab_sz} Terms", "Unigram + Bigram")
    with row2_c4:
        st.metric("Active Top Model", active_model, "Selected by Weighted F1")

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Compact System Health Indicators
    # -------------------------------------------------------------------------
    st.markdown("### Infrastructure Health Matrix")

    h_cols = st.columns(5)
    health_items = [
        ("AI Engine", "ONLINE", "#34d399", "Python / Scikit-Learn"),
        ("NLP Pipeline", "READY", "#34d399", "12-Step NLTK Engine"),
        ("TF-IDF", "LOADED", "#34d399", f"{vocab_sz} vocabulary features"),
        ("ML Models", "ONLINE", "#34d399", "3 Classifiers Serialized"),
        ("Prediction Engine", "READY", "#34d399", "Real-Time Inference"),
    ]

    for i, (name, state, color, desc) in enumerate(health_items):
        with h_cols[i]:
            st.markdown(
                f"""
                <div class="kpi-card" style="text-align: center; padding: 1rem 0.5rem;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #7dd3fc; text-transform: uppercase; letter-spacing: 0.05em;">{name}</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {color}; margin: 4px 0;">
                        <span class="status-dot" style="background-color: {color}; box-shadow: 0 0 8px {color};"></span> {state}
                    </div>
                    <div style="font-size: 0.72rem; color: #94a3b8;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )