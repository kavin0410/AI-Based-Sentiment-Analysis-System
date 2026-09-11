"""Analyzer Component for AI Sentiment Intelligence.

Provides the primary SaaS sentiment analyzer experience:
- Large text area with character counter (0 / 5000)
- 6 quick example pills (Very Positive, Positive, Neutral, Negative, Very Negative, Mixed)
- Primary CTA: "✨ Analyze Sentiment"
- Prediction output: Sentiment, Confidence, Probabilities, Model, Timestamp
- Technical explainability expander: Original text, Cleaned text, NLP process, TF-IDF representation
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
    """Cache loaded best model and vectorizer."""
    model, model_name, metadata = load_best_model()
    vectorizer = load_vectorizer()
    return model, model_name, metadata, vectorizer


def render_analyze_page():
    """Render the primary Sentiment Analyzer page."""
    st.markdown("## ✨ Sentiment Analyzer")
    st.caption("Classify unstructured text in real time with confidence scoring and step-by-step pipeline inspection.")

    val_report = validate_model_artifacts()
    if not val_report["valid"]:
        st.error("Model artifacts are missing. Please ensure models are trained.")
        return

    try:
        model, model_name, meta, vectorizer = get_cached_prediction_artifacts()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # -------------------------------------------------------------------------
    # 1. Example Presets
    # -------------------------------------------------------------------------
    st.markdown("##### Quick Presets")
    preset_dict = {
        "Very Positive": "I am absolutely blown away by how extraordinary and perfect this product is! Best purchase ever.",
        "Positive": "Customer support was fast, friendly, and solved my issue within minutes.",
        "Neutral": "The shipment arrived on Wednesday afternoon. The package includes three replacement cables.",
        "Negative": "The battery life is noticeably worse than advertised and drains in under three hours.",
        "Very Negative": "Completely useless and broke on day one! Customer service refused to refund my money.",
        "Mixed": "I didn't hate the device, but the interface was unintuitive and confusing at first.",
    }

    cols = st.columns(6)
    for i, (label, text_val) in enumerate(preset_dict.items()):
        with cols[i]:
            if st.button(label, use_container_width=True, key=f"btn_preset_{i}"):
                st.session_state["analyzer_input_text"] = text_val

    # -------------------------------------------------------------------------
    # 2. Text Input Area with Character Count
    # -------------------------------------------------------------------------
    st.markdown("##### Input Text")
    current_val = st.session_state.get("analyzer_input_text", "")
    input_text = st.text_area(
        label="Input text to analyze:",
        value=current_val,
        height=140,
        max_chars=config.MAX_INPUT_LENGTH,
        placeholder="Type or paste any text review, feedback, or message here...",
        key="analyzer_main_textarea",
        label_visibility="collapsed",
    )

    char_cnt = len(input_text)
    st.caption(f"**{char_cnt}** / {config.MAX_INPUT_LENGTH} characters")

    col_b1, col_b2, _ = st.columns([2, 1, 4])
    with col_b1:
        btn_analyze = st.button("✨ Analyze Sentiment", type="primary", use_container_width=True, key="btn_run_analyzer")
    with col_b2:
        if st.button("Clear", use_container_width=True, key="btn_clear_analyzer"):
            st.session_state["analyzer_input_text"] = ""
            st.rerun()

    # -------------------------------------------------------------------------
    # 3. Prediction Output
    # -------------------------------------------------------------------------
    result = None
    if btn_analyze and input_text.strip():
        with st.spinner("Analyzing text with machine learning engine..."):
            result = predict_sentiment(
                text=input_text,
                model=model,
                vectorizer=vectorizer,
                model_name=model_name,
            )
            st.session_state["analyzer_last_result"] = result
            history_mgr = st.session_state.get("history_manager")
            if history_mgr and result.get("valid"):
                history_mgr.add_prediction(result)
    elif "analyzer_last_result" in st.session_state:
        result = st.session_state["analyzer_last_result"]

    if result and result.get("valid"):
        st.divider()
        st.markdown("### Prediction Result")

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
                <div class="asi-card">
                    <div style="font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Polarity</div>
                    <div style="margin-top: 8px;">
                        <span class="{badge_class}" style="font-size: 1.4rem; padding: 8px 20px;">
                            {sentiment.upper()}
                        </span>
                    </div>
                    <div style="margin-top: 16px; font-size: 0.85rem; color: #8b949e;">
                        Model: <strong style="color: #f0f6fc;">{m_used}</strong>
                    </div>
                    <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px;">
                        Analyzed at: {timestamp[:19].replace('T', ' ')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if score_type == "probability":
                st.metric("Confidence", f"{score * 100:.1f}%")
            else:
                st.metric("Decision Score", f"{score:.4f}")
                st.caption("Raw decision score (Linear SVM) — not a probability.")

        with col_res2:
            st.markdown("##### Probability Distribution")
            if score_type == "probability":
                for cls in ["Positive", "Neutral", "Negative"]:
                    p_val = probs.get(cls, 0.0)
                    st.write(f"**{cls}:** {p_val * 100:.1f}%")
                    st.progress(min(max(p_val, 0.0), 1.0))
            else:
                st.caption("Decision Scores per Class:")
                for cls in ["Negative", "Neutral", "Positive"]:
                    st.write(f"**{cls}:** `{probs.get(cls, 0.0):.4f}`")

        # ---------------------------------------------------------------------
        # 4. Explainability Expander
        # ---------------------------------------------------------------------
        with st.expander("🔍 How the AI processed this text", expanded=False):
            st.markdown("**1. Original Text**")
            st.code(result.get("text", input_text), language="text")

            st.markdown("**2. Cleaned & Preprocessed Text (TF-IDF Input)**")
            st.code(result.get("processed_text", ""), language="text")

            st.markdown("**3. NLP Processing Pipeline**")
            st.markdown(
                """
                - Contractions expanded (`don't` → `do not`, `didn't` → `did not`)
                - Normalized to lowercase & special characters stripped
                - Tokenized with NLTK tokenizer
                - Negation words (`not`, `no`, `never`) preserved
                - Tokens lemmatized to base roots via WordNetLemmatizer
                """
            )

            st.markdown("**4. TF-IDF Representation**")
            st.caption(f"Vectorized against pre-fitted {result.get('vocab_size', 570)}-term vocabulary (unigrams + bigrams, sublinear TF scaling).")