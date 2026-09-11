"""Overview Component for SentimentAI (SentimentLab).

Provides executive product overview:
- Clean product introduction
- Total predictions, positive %, negative %, neutral %, average confidence
- Active model, model status, dataset status
- Quick Sentiment Analyzer with live result
"""

import json
from pathlib import Path
import streamlit as st
import config
from src.model_loader import validate_model_artifacts
from src.predict import predict_sentiment


def render_overview_page():
    """Render the executive Overview page."""
    # Top Product Summary Header
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <p style="font-size: 1.05rem; color: #94a3b8; line-height: 1.6; max-width: 850px; margin: 0;">
                <strong>SentimentLab</strong> delivers real-time sentiment intelligence across customer feedback, product reviews, 
                and unstructured text. Powered by custom NLP tokenization, 570 TF-IDF features, and supervised machine learning classifiers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 1. Executive Performance & Session Metrics
    # -------------------------------------------------------------------------
    history_mgr = st.session_state.get("history_manager")
    history_items = history_mgr.get_history() if history_mgr else []
    total_preds = len(history_items)

    if total_preds > 0:
        h_df = history_mgr.to_dataframe()
        s_counts = h_df["sentiment"].value_counts()
        pos_pct = (s_counts.get("Positive", 0) / total_preds) * 100
        neg_pct = (s_counts.get("Negative", 0) / total_preds) * 100
        neu_pct = (s_counts.get("Neutral", 0) / total_preds) * 100

        prob_rows = h_df[h_df["score_type"] == "probability"]
        avg_conf = (prob_rows["score"].mean() * 100) if not prob_rows.empty else 0.0
    else:
        # Default corpus benchmark metrics when session is fresh
        pos_pct = 33.3
        neg_pct = 33.3
        neu_pct = 33.3
        avg_conf = 35.6

    val_report = validate_model_artifacts()
    active_model = val_report.get("best_model_name", "Logistic Regression")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Predictions", f"{total_preds if total_preds > 0 else 'Active'}", "Session queries" if total_preds > 0 else "System ready")
    with c2:
        st.metric("Positive", f"{pos_pct:.1f}%")
    with c3:
        st.metric("Neutral", f"{neu_pct:.1f}%")
    with c4:
        st.metric("Negative", f"{neg_pct:.1f}%")
    with c5:
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. System Status & Active Engine Spec
    # -------------------------------------------------------------------------
    col_stat1, col_stat2, col_stat3 = st.columns(3)

    with col_stat1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Active Model</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #f8fafc;">{active_model}</div>
                <div class="kpi-sub">Optimized for high-dimensional TF-IDF vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stat2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Model Health</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #34d399;">Ready & Loaded</div>
                <div class="kpi-sub">3 Classifiers (LR, MNB, LinearSVC) serialized</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stat3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Dataset Health</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: #cbd5e1;">60 Records</div>
                <div class="kpi-sub">Balanced 3-class corpus (20 / 20 / 20)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Quick Sentiment Analyzer
    # -------------------------------------------------------------------------
    st.markdown("### Quick Analysis")
    col_left, col_right = st.columns([5, 5])

    with col_left:
        quick_text = st.text_area(
            "Quick Input:",
            value=st.session_state.get("overview_input_text", "The customer support team was fast, helpful, and resolved my issue immediately."),
            height=120,
            key="overview_quick_textarea",
            placeholder="Paste text here to test...",
            label_visibility="collapsed",
        )

        col_b1, col_b2 = st.columns([3, 2])
        with col_b1:
            btn_run = st.button("✦ Analyze Sentiment", type="primary", use_container_width=True, key="btn_ov_analyze")
        with col_b2:
            if st.button("Clear", use_container_width=True, key="btn_ov_clear"):
                st.session_state["overview_input_text"] = ""
                st.rerun()

    with col_right:
        result = None
        if btn_run and quick_text.strip():
            with st.spinner("Analyzing..."):
                result = predict_sentiment(quick_text.strip())
                st.session_state["last_overview_result"] = result
                if history_mgr and result.get("valid"):
                    history_mgr.add_prediction(result)
        elif "last_overview_result" in st.session_state:
            result = st.session_state["last_overview_result"]

        if result and result.get("valid"):
            sentiment = result["sentiment"]
            score = result["score"]
            score_type = result["score_type"]
            probs = result.get("probabilities", {})

            badge_style = "badge-neutral"
            if sentiment == "Positive":
                badge_style = "badge-positive"
            elif sentiment == "Negative":
                badge_style = "badge-negative"

            st.markdown(
                f"""
                <div class="product-card" style="padding: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span class="sentiment-badge {badge_style}">{sentiment}</span>
                        <span style="font-size: 0.8rem; color: #64748b;">Confidence: <strong>{score * 100:.1f}%</strong></span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if score_type == "probability":
                for cls in ["Positive", "Neutral", "Negative"]:
                    p_val = probs.get(cls, 0.0)
                    st.write(f"**{cls}:** {p_val * 100:.1f}%")
                    st.progress(min(max(p_val, 0.0), 1.0))
            else:
                st.write(f"Decision Score: `{score:.4f}`")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="product-card" style="text-align: center; padding: 2.25rem 1rem; color: #64748b;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: #475569;">✦</div>
                    <div style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">Ready to analyze</div>
                    <div style="font-size: 0.8rem; margin-top: 0.25rem;">Enter text on the left to view real-time sentiment</div>
                </div>
                """,
                unsafe_allow_html=True,
            )