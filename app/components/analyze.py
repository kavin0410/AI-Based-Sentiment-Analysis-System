"""Analyze Component for SentimentAI (SentimentLab).

The primary interaction workspace:
- Large text area with intuitive placeholder: "Paste a review, comment, feedback, or message..."
- Primary button: "Analyze Sentiment"
- Results: Sentiment, Confidence, Positive/Neutral/Negative probabilities, Model, Timestamp
- Expandable: "How the AI analyzed this text"
  Text -> Cleaning -> NLP -> TF-IDF -> ML Model -> Sentiment
"""

from datetime import datetime
import streamlit as st
import config
from src.model_loader import (
    load_best_model,
    load_vectorizer,
    validate_model_artifacts,
)
from src.predict import predict_sentiment


@st.cache_resource
def get_cached_prediction_artifacts():
    """Load and cache best model and vectorizer in memory."""
    model, model_name, metadata = load_best_model()
    vectorizer = load_vectorizer()
    return model, model_name, metadata, vectorizer


def render_analyze_page():
    """Render the primary Sentiment Analyzer workspace."""
    st.markdown("## ✦ Analyze")
    st.caption("Classify textual feedback in real time with probabilistic confidence scores and transparent explainability.")

    val_report = validate_model_artifacts()
    if not val_report["valid"]:
        st.error("Model artifacts are not currently accessible.")
        return

    try:
        model, model_name, meta, vectorizer = get_cached_prediction_artifacts()
    except Exception as e:
        st.error(f"Error initializing prediction engine: {e}")
        return

    # -------------------------------------------------------------------------
    # 1. Quick Example Pills
    # -------------------------------------------------------------------------
    presets = {
        "Positive Feedback": "The product quality exceeded my expectations. Smooth setup, beautiful design, and very responsive customer support.",
        "Neutral Review": "The package arrived on Thursday as scheduled. It includes the charging adapter, cable, and user manual.",
        "Negative Criticism": "The device frequently disconnects from Wi-Fi and the battery drains completely within three hours. Very disappointed.",
        "Nuanced Review": "I didn't think the application was bad, but the onboarding workflow was somewhat confusing.",
    }

    st.markdown("##### Quick Presets")
    p_cols = st.columns(4)
    for i, (p_label, p_text) in enumerate(presets.items()):
        with p_cols[i]:
            if st.button(p_label, use_container_width=True, key=f"btn_p_{i}"):
                st.session_state["analyze_active_text"] = p_text

    # -------------------------------------------------------------------------
    # 2. Main Textarea & Primary Action
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    current_text = st.session_state.get("analyze_active_text", "")
    input_text = st.text_area(
        label="Input Text",
        value=current_text,
        height=150,
        max_chars=config.MAX_INPUT_LENGTH,
        placeholder="Paste a review, comment, feedback, or message...",
        key="main_sentiment_textarea",
        label_visibility="collapsed",
    )

    char_cnt = len(input_text)
    col_c1, col_c2 = st.columns([4, 1])
    with col_c1:
        st.caption(f"{char_cnt} / {config.MAX_INPUT_LENGTH} characters")

    col_btn, col_clear, _ = st.columns([3, 1, 4])
    with col_btn:
        btn_run = st.button("Analyze Sentiment", type="primary", use_container_width=True, key="btn_run_main_analysis")
    with col_clear:
        if st.button("Clear", use_container_width=True, key="btn_clear_main_analysis"):
            st.session_state["analyze_active_text"] = ""
            st.rerun()

    # -------------------------------------------------------------------------
    # 3. Sentiment Results Card
    # -------------------------------------------------------------------------
    result = None
    if btn_run and input_text.strip():
        with st.spinner("Analyzing text through neural NLP pipeline..."):
            result = predict_sentiment(
                text=input_text,
                model=model,
                vectorizer=vectorizer,
                model_name=model_name,
            )
            st.session_state["last_analysis_result"] = result
            history_mgr = st.session_state.get("history_manager")
            if history_mgr and result.get("valid"):
                history_mgr.add_prediction(result)
    elif "last_analysis_result" in st.session_state:
        result = st.session_state["last_analysis_result"]

    if result and result.get("valid"):
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        sentiment = result["sentiment"]
        score = result["score"]
        score_type = result["score_type"]
        m_used = result.get("model_name", model_name)
        timestamp = result.get("timestamp", datetime.now().isoformat())
        probs = result.get("probabilities", {})

        badge_class = "badge-neutral"
        if sentiment == "Positive":
            badge_class = "badge-positive"
        elif sentiment == "Negative":
            badge_class = "badge-negative"

        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.markdown(
                f"""
                <div class="product-card">
                    <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                        Classification
                    </div>
                    <div style="margin: 0.5rem 0 1rem 0;">
                        <span class="sentiment-badge {badge_class}" style="font-size: 1.35rem; padding: 6px 20px;">
                            {sentiment}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.7;">
                        Confidence: <strong style="color: #f8fafc;">{score * 100:.1f}%</strong><br/>
                        Model: <strong style="color: #cbd5e1;">{m_used}</strong><br/>
                        Timestamp: <span style="color: #64748b;">{timestamp[:19].replace('T', ' ')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_res2:
            st.markdown(
                """
                <div class="product-card">
                    <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
                        Probability Distribution
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
                for cls in ["Negative", "Neutral", "Positive"]:
                    st.write(f"**{cls}:** `{probs.get(cls, 0.0):.4f}` (Decision score)")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 4. Expandable: How the AI Analyzed this Text
        # ---------------------------------------------------------------------
        with st.expander("How the AI analyzed this text", expanded=False):
            st.markdown(
                """
                <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 1rem; color: #64748b; font-size: 0.85rem;">
                    <span>Input Text</span> → <span>Cleaning</span> → <span>NLP</span> → <span>TF-IDF</span> → <span>ML Model</span> → <span style="color: #388bfd; font-weight: 600;">Sentiment</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.markdown("**Original Text**")
                st.code(result.get("text", input_text), language="text")

            with col_exp2:
                st.markdown("**Cleaned & Tokenized Text**")
                st.code(result.get("processed_text", ""), language="text")

            st.markdown(
                f"""
                - **Preprocessing:** Contraction expansion, lowercase normalization, special character stripping.
                - **Negation Handling:** Negative tokens (*not, no, never*) were preserved to safeguard polarity.
                - **Feature Extraction:** Transformed across **{result.get('vocab_size', 570)}** pre-fitted TF-IDF n-gram features.
                - **Inference Engine:** Evaluated by **{m_used}**.
                """
            )