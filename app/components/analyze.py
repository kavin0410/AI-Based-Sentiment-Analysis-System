"""Analyze Page Component for AI Sentiment Intelligence (Stage 6).

Provides the primary interactive real-time sentiment analyzer:
- Input text area with live character counter
- Clickable example sentences
- Real-time sentiment classification using the evaluated best model
- Score & probability breakdown display
- Step-by-step NLP transformation inspector
- Technical prediction metadata inspector
- Session prediction history table with CSV export
"""

from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.model_loader import (
    load_best_model,
    load_vectorizer,
    validate_model_artifacts,
)
from src.predict import (
    PredictionHistoryManager,
    format_prediction_result,
    predict_sentiment,
)


@st.cache_resource
def get_cached_prediction_artifacts():
    """Cache loaded best model and vectorizer in Streamlit memory."""
    model, model_name, metadata = load_best_model()
    vectorizer = load_vectorizer()
    return model, model_name, metadata, vectorizer


def render_analyze_page():
    """Render the primary Real-Time Sentiment Analyzer page."""
    st.markdown("## ⚡ Real-Time Sentiment Engine")
    st.caption("Classify new, unseen text using the dynamically selected best ML model and TF-IDF feature extractor.")

    # 1. Artifact Validation Check
    val_report = validate_model_artifacts()
    if not val_report["valid"]:
        st.error("🚨 Model Artifact Validation Failed!")
        for err in val_report.get("errors", []):
            st.warning(f"- {err}")
        st.info("Please execute Stage 3 (`python run_stage3.py`) and Stage 4 (`python run_stage4.py`) first.")
        return

    # 2. Load cached model & vectorizer
    try:
        model, model_name, meta, vectorizer = get_cached_prediction_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return

    # Active Best Model Info Banner
    best_f1 = meta.get("f1_score", 0.0)
    st.info(
        f"🤖 **Active Best Model:** `{model_name}` | "
        f"**Selection Criterion:** Weighted F1 Score ({best_f1*100:.2f}%) | "
        f"**TF-IDF Vocabulary:** {len(vectorizer.get_feature_names_out())} features"
    )

    # 3. Example Sentences Selector
    st.markdown("##### 💡 Try an Example Sentence")
    example_sentences = [
        "I absolutely love this product! It works perfectly and exceeded all my expectations.",
        "This is by far the worst experience I have ever had; complete waste of money.",
        "The product arrived yesterday on time and the package contains standard accessories.",
        "Customer support was quick, friendly, and resolved my issue in minutes.",
        "I wouldn't recommend this service to anyone; they refused to issue a refund.",
        "I didn't think the movie was bad, but the pacing was somewhat slow.",
    ]

    example_cols = st.columns(3)
    for i, ex in enumerate(example_sentences):
        with example_cols[i % 3]:
            if st.button(f"Try Example {i+1}", key=f"ex_analyze_btn_{i}", use_container_width=True):
                st.session_state["user_text_input"] = ex

    # 4. Text Input Area
    st.divider()
    st.markdown("##### 📝 Enter Custom Text to Analyze")

    input_text = st.text_area(
        "Input Text:",
        value=st.session_state.get("user_text_input", ""),
        height=120,
        max_chars=config.MAX_INPUT_LENGTH,
        placeholder="Type or paste any text review, comment, or message here...",
        key="main_text_input_area",
    )

    char_count = len(input_text)
    st.caption(f"Character Count: **{char_count} / {config.MAX_INPUT_LENGTH}**")

    # Action Buttons
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 6])
    with col_btn1:
        btn_analyze = st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True, key="btn_main_analyze")
    with col_btn2:
        if st.button("🗑️ Clear Input", use_container_width=True, key="btn_clear_input"):
            st.session_state["user_text_input"] = ""
            st.rerun()

    # 5. Execute Prediction when button clicked
    if btn_analyze:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment using NLP engine & ML model..."):
                result = predict_sentiment(
                    text=input_text,
                    model=model,
                    vectorizer=vectorizer,
                    model_name=model_name,
                )

            if not result.get("valid", False):
                st.error(f"Validation Error: {result.get('error')}")
            else:
                # Add to session history
                history_mgr = st.session_state.get("history_manager")
                if history_mgr:
                    history_mgr.add_prediction(result)

                # Render Result Card
                st.divider()
                st.markdown("### 🏁 Prediction Output")

                sentiment = result["sentiment"]
                score = result["score"]
                score_type = result["score_type"]

                col_res1, col_res2 = st.columns([1, 1])

                with col_res1:
                    # Visual Card styling
                    if sentiment == "Positive":
                        st.success(f"### 🟢 POSITIVE\nPredicted Sentiment Category")
                    elif sentiment == "Negative":
                        st.error(f"### 🔴 NEGATIVE\nPredicted Sentiment Category")
                    else:
                        st.info(f"### ⚪ NEUTRAL\nPredicted Sentiment Category")

                    st.markdown(f"**Model Used:** `{model_name}`")

                    if score_type == "probability":
                        st.metric("Confidence Score", f"{score * 100:.1f}%")
                    else:
                        st.metric("Decision Score", f"{score:.4f}")
                        st.caption("ℹ️ *Decision score is the model's classification score and is not a probability.*")

                with col_res2:
                    st.markdown("##### 📊 Class Score Breakdown")
                    probs = result.get("probabilities", {})

                    if score_type == "probability":
                        for cls in ["Positive", "Neutral", "Negative"]:
                            p_val = probs.get(cls, 0.0)
                            st.write(f"**{cls}:** {p_val*100:.1f}%")
                            st.progress(min(max(p_val, 0.0), 1.0))
                    else:
                        st.caption("Decision Scores per Class (Linear SVM):")
                        for cls in ["Negative", "Neutral", "Positive"]:
                            sc_val = probs.get(cls, 0.0)
                            st.write(f"**{cls}:** `{sc_val:.4f}`")

                # Expandable Preprocessing Details
                with st.expander("🔍 View NLP Preprocessing Pipeline Transformation", expanded=False):
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        st.info(f"**Original Input Text:**\n\n{result['text']}")
                    with col_exp2:
                        st.success(f"**Preprocessed Clean Text (TF-IDF Input):**\n\n`{result['processed_text']}`")

                # Model Metadata Details
                with st.expander("📋 Prediction Technical Metadata", expanded=False):
                    st.json({
                        "selected_model": result["model_name"],
                        "score_type": result["score_type"],
                        "score_value": result["score"],
                        "tfidf_features": result["vocab_size"],
                        "timestamp": result["timestamp"],
                    })

    # 6. Session Prediction History Table
    st.divider()
    st.markdown("### 📜 Session Prediction History")
    history_mgr = st.session_state.get("history_manager")

    if not history_mgr or not history_mgr.get_history():
        st.info("No predictions recorded in this session yet. Enter text above to analyze.")
    else:
        history_df = history_mgr.to_dataframe()
        st.caption(f"Showing last **{len(history_df)} / {config.PREDICTION_HISTORY_LIMIT}** session predictions.")

        display_history = history_df.copy()
        display_history["text_preview"] = display_history["text"].apply(
            lambda t: t[:60] + "..." if len(t) > 60 else t
        )
        display_history["score_display"] = display_history.apply(
            lambda r: f"{r['score']*100:.1f}%" if r["score_type"] == "probability" else f"{r['score']:.4f}",
            axis=1,
        )

        st.dataframe(
            display_history[["timestamp", "text_preview", "sentiment", "model", "score_display"]].sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        col_hist1, col_hist2 = st.columns([3, 2])
        with col_hist1:
            csv_data = history_mgr.to_csv()
            st.download_button(
                label="📥 Download Session History (CSV)",
                data=csv_data,
                file_name=f"sentiment_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download_csv_analyze",
            )
        with col_hist2:
            if st.button("🗑️ Clear Session History", use_container_width=True, key="btn_clear_history_analyze"):
                history_mgr.clear()
                st.rerun()
