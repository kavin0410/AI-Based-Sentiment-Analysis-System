"""Analyze Component for SentimentLab.

The primary product experience:
1. Single Text Analysis:
   - "Paste a review, comment, feedback, message, or any text..."
   - Character counter: 0 / 5000
   - Example buttons: Positive, Negative, Neutral, Mixed, Negation
   - Primary CTA: Analyze Sentiment
   - Result panel: SENTIMENT, CONFIDENCE, Probability distribution (Pos, Neu, Neg), Model, Timestamp
   - Sentiment Meter: Negative ─────────●──────── Positive
   - "Analyze Another" button
   - Explainability: "How SentimentLab reached this result"
     Input Text -> Data Cleaning -> NLP Preprocessing -> TF-IDF Feature Extraction -> ML Model -> Sentiment Prediction
     Original, Cleaned, Tokenized, Stopwords, Negation preservation, Lemmatized, TF-IDF

2. Text Comparison:
   - Text A and Text B independent analysis
   - Comparison summary

3. Batch Analysis (CSV Upload):
   - Upload CSV with "text" column
   - Validate, preview, run batch prediction with progress bar
   - Display results table, filter, and CSV download
"""

from datetime import datetime
import io
import pandas as pd
import streamlit as st
import config
from src.model_loader import (
    load_best_model,
    load_vectorizer,
    validate_model_artifacts,
)
from src.predict import predict_batch, predict_sentiment
from src.preprocessing import (
    expand_contractions,
    lemmatize_tokens,
    remove_stopwords,
    tokenize_text,
)
from src.data_cleaning import clean_text_basic


@st.cache_resource
def get_cached_prediction_artifacts():
    """Cache loaded best model and vectorizer."""
    model, model_name, metadata = load_best_model()
    vectorizer = load_vectorizer()
    return model, model_name, metadata, vectorizer


def render_analyze_page():
    """Render the primary SentimentLab Analyzer experience with sub-modes."""
    st.markdown("## ✦ Analyze Workspace")
    st.caption("Perform real-time natural language classification, side-by-side text comparisons, or large batch file inference.")

    val_report = validate_model_artifacts()
    if not val_report["valid"]:
        st.error("Model artifacts are currently unavailable. Ensure models are trained.")
        return

    try:
        model, model_name, meta, vectorizer = get_cached_prediction_artifacts()
    except Exception as e:
        st.error(f"Error loading inference engine: {e}")
        return

    # Sub-mode selection
    sub_mode = st.radio(
        "Analysis Mode",
        ["Single Text Analysis", "Text Comparison", "Batch CSV Analysis"],
        horizontal=True,
        key="analyze_submode_radio",
        label_visibility="collapsed",
    )
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if sub_mode == "Single Text Analysis":
        _render_single_text_analysis(model, model_name, vectorizer)
    elif sub_mode == "Text Comparison":
        _render_text_comparison(model, model_name, vectorizer)
    elif sub_mode == "Batch CSV Analysis":
        _render_batch_analysis(model, model_name, vectorizer)


def _render_single_text_analysis(model, model_name, vectorizer):
    """Render the core Single Text Analyzer."""
    # 1. Preset Buttons
    st.markdown("##### Quick Presets")
    presets = {
        "Positive": "The build quality is extraordinary, setup took less than two minutes, and customer support was outstanding!",
        "Negative": "Terrible customer service. The device arrived damaged and the company completely refused to issue a refund.",
        "Neutral": "The shipment arrived on Wednesday afternoon. It contains three charging cables and standard instructions.",
        "Mixed": "The screen display is breathtakingly vibrant, but the battery drains in under three hours.",
        "Negation": "I didn't think the performance would be bad, but it isn't quite as fast as advertised.",
    }

    cols = st.columns(5)
    for i, (p_name, p_text) in enumerate(presets.items()):
        with cols[i]:
            if st.button(p_name, use_container_width=True, key=f"btn_p_{p_name}"):
                st.session_state["single_text_input_val"] = p_text

    # 2. Text Input & Counter
    current_val = st.session_state.get("single_text_input_val", "")
    input_text = st.text_area(
        label="Input Text",
        value=current_val,
        height=145,
        max_chars=config.MAX_INPUT_LENGTH,
        placeholder="Paste a review, comment, feedback, message, or any text...",
        key="analyzer_main_textarea_v2",
        label_visibility="collapsed",
    )

    char_cnt = len(input_text)
    st.caption(f"{char_cnt} / {config.MAX_INPUT_LENGTH}")

    # Action Buttons
    col_run, col_clear, _ = st.columns([3, 1, 4])
    with col_run:
        btn_analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True, key="btn_run_single")
    with col_clear:
        if st.button("Clear", use_container_width=True, key="btn_clear_single"):
            st.session_state["single_text_input_val"] = ""
            st.rerun()

    # 3. Execution & Result Panel
    result = None
    if btn_analyze and input_text.strip():
        with st.spinner("Processing text..."):
            result = predict_sentiment(
                text=input_text,
                model=model,
                vectorizer=vectorizer,
                model_name=model_name,
            )
            st.session_state["last_single_result"] = result
            history_mgr = st.session_state.get("history_manager")
            if history_mgr and result.get("valid"):
                history_mgr.add_prediction(result)
            st.toast("Analysis completed successfully!", icon="✅")
    elif "last_single_result" in st.session_state:
        result = st.session_state["last_single_result"]

    if result and result.get("valid"):
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### Analysis Result")

        sentiment = result["sentiment"]
        score = result["score"]
        score_type = result["score_type"]
        m_used = result.get("model_name", model_name)
        timestamp = result.get("timestamp", datetime.now().isoformat())
        probs = result.get("probabilities", {})

        pos_p = probs.get("Positive", 0.0)
        neu_p = probs.get("Neutral", 0.0)
        neg_p = probs.get("Negative", 0.0)

        badge_class = "badge-neutral"
        marker_left_pct = 50
        if sentiment == "Positive":
            badge_class = "badge-positive"
            marker_left_pct = 50 + int(score * 45)
        elif sentiment == "Negative":
            badge_class = "badge-negative"
            marker_left_pct = 50 - int(score * 45)

        col_card1, col_card2 = st.columns([1, 1])

        with col_card1:
            st.markdown(
                f"""
                <div class="product-card">
                    <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #7dd3fc; letter-spacing: 0.08em; margin-bottom: 0.5rem;">
                        SENTIMENT CLASSIFICATION
                    </div>
                    <div style="margin: 0.5rem 0 1.25rem 0;">
                        <span class="sentiment-badge {badge_class}" style="font-size: 1.4rem; padding: 8px 24px;">
                            {sentiment}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.8;">
                        Confidence: <strong style="color: #ffffff; font-size: 1.1rem;">{score * 100:.1f}%</strong><br/>
                        Model: <strong style="color: #bae6fd;">{m_used}</strong><br/>
                        Timestamp: <span style="color: #64748b;">{timestamp[:19].replace('T', ' ')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_card2:
            st.markdown(
                """
                <div class="product-card">
                    <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #7dd3fc; letter-spacing: 0.08em; margin-bottom: 0.75rem;">
                        PROBABILITY DISTRIBUTION
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if score_type == "probability":
                for cls, p_val in [("Positive", pos_p), ("Neutral", neu_p), ("Negative", neg_p)]:
                    st.write(f"**{cls}:** {p_val * 100:.1f}%")
                    st.progress(min(max(p_val, 0.0), 1.0))
            else:
                for cls in ["Negative", "Neutral", "Positive"]:
                    st.write(f"**{cls}:** `{probs.get(cls, 0.0):.4f}` (Decision Score)")
            st.markdown("</div>", unsafe_allow_html=True)

        # Sentiment Meter
        st.markdown(
            f"""
            <div class="meter-container">
                <div class="meter-labels">
                    <span style="color: #f87171;">Negative</span>
                    <span style="color: #7dd3fc;">Neutral</span>
                    <span style="color: #34d399;">Positive</span>
                </div>
                <div class="meter-track">
                    <div class="meter-marker" style="left: {marker_left_pct}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Analyze Another Button
        if st.button("✦ Analyze Another Text", use_container_width=True, key="btn_analyze_another"):
            st.session_state["single_text_input_val"] = ""
            st.session_state.pop("last_single_result", None)
            st.rerun()

        # 4. Explainability Expander: "How SentimentLab reached this result"
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        with st.expander("How SentimentLab reached this result", expanded=False):
            st.markdown(
                """
                <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 1.25rem; color: #94a3b8; font-size: 0.85rem; flex-wrap: wrap;">
                    <span>Input Text</span> → <span>Data Cleaning</span> → <span>NLP Preprocessing</span> → <span>TF-IDF Feature Extraction</span> → <span>ML Model</span> → <span style="color: #38bdf8; font-weight: 700;">Sentiment Prediction</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            raw_t = result.get("text", input_text)
            s1_exp = expand_contractions(raw_t)
            s2_clean = clean_text_basic(s1_exp)
            s3_tok = tokenize_text(s2_clean)
            s4_stop = remove_stopwords(s3_tok, preserve_negations=True)
            s5_lem = lemmatize_tokens(s4_stop)
            s6_final = result.get("processed_text", " ".join(s5_lem))

            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                st.markdown("**Original Text**")
                st.code(raw_t, language="text")
                st.markdown("**Cleaned & Normalized Text**")
                st.code(s2_clean, language="text")
                st.markdown("**Tokenized Tokens**")
                st.write(s3_tok)

            with c_exp2:
                st.markdown("**Stopword Filtering (Negations Preserved)**")
                st.write(s4_stop)
                st.markdown("**Lemmatized Tokens (WordNet)**")
                st.write(s5_lem)
                st.markdown("**Final Processed Text (TF-IDF Input)**")
                st.code(s6_final, language="text")

            st.caption(f"Vectorized into {result.get('vocab_size', 570)} TF-IDF n-gram dimensions and classified by {m_used}.")


def _render_text_comparison(model, model_name, vectorizer):
    """Render side-by-side text comparison workspace."""
    st.markdown("### Text Comparison Tool")
    st.caption("Compare sentiment polarity, confidence distributions, and key phrase reactions across two independent texts.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Text A")
        text_a = st.text_area("Input Text A", value="The camera quality on this new phone is breathtaking and works seamlessly.", height=120, key="cmp_text_a")
    with col_b:
        st.markdown("##### Text B")
        text_b = st.text_area("Input Text B", value="The battery life is frustratingly short and overheats after 20 minutes of usage.", height=120, key="cmp_text_b")

    if st.button("Compare Texts", type="primary", use_container_width=True, key="btn_run_compare"):
        if not text_a.strip() or not text_b.strip():
            st.warning("Please enter text in both Text A and Text B to compare.")
            return

        with st.spinner("Analyzing both texts..."):
            res_a = predict_sentiment(text_a, model=model, vectorizer=vectorizer, model_name=model_name)
            res_b = predict_sentiment(text_b, model=model, vectorizer=vectorizer, model_name=model_name)

        col_ra, col_rb = st.columns(2)
        for target_col, res, label in [(col_ra, res_a, "Text A"), (col_rb, res_b, "Text B")]:
            with target_col:
                sent = res["sentiment"]
                score = res["score"]
                probs = res.get("probabilities", {})
                badge = "badge-positive" if sent == "Positive" else ("badge-negative" if sent == "Negative" else "badge-neutral")

                st.markdown(
                    f"""
                    <div class="product-card">
                        <div style="font-weight: 700; color: #7dd3fc; font-size: 0.85rem;">{label} RESULT</div>
                        <div style="margin: 0.5rem 0;">
                            <span class="sentiment-badge {badge}">{sent}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">
                            Confidence: <strong style="color: #ffffff;">{score*100:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if res["score_type"] == "probability":
                    for cls in ["Positive", "Neutral", "Negative"]:
                        st.write(f"**{cls}:** {probs.get(cls, 0.0)*100:.1f}%")
                        st.progress(min(max(probs.get(cls, 0.0), 0.0), 1.0))

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.info(f"**Comparison Summary:** Text A classified as **{res_a['sentiment']}** ({res_a['score']*100:.1f}%), while Text B classified as **{res_b['sentiment']}** ({res_b['score']*100:.1f}%).")


def _render_batch_analysis(model, model_name, vectorizer):
    """Render batch CSV file analysis."""
    st.markdown("### Batch CSV Analysis")
    st.caption("Upload a CSV file containing a `text` column to execute high-throughput batch sentiment classification.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="batch_csv_uploader")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse CSV file: {e}")
            return

        if "text" not in df.columns:
            st.error("Invalid CSV structure. The file must contain a 'text' column.")
            return

        st.success(f"CSV uploaded successfully! Found **{len(df)}** records.")
        st.caption("Preview of uploaded records (first 5):")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary", key="btn_run_batch"):
            texts_list = df["text"].dropna().astype(str).tolist()

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            status_text.text(f"Classifying {len(texts_list)} texts...")
            results = []

            for idx, txt in enumerate(texts_list):
                res = predict_sentiment(txt, model=model, vectorizer=vectorizer, model_name=model_name)
                results.append({
                    "Text": txt,
                    "Predicted Sentiment": res.get("sentiment", "Unknown"),
                    "Confidence": f"{res.get('score', 0.0)*100:.1f}%" if res.get("score_type") == "probability" else f"{res.get('score', 0.0):.4f}",
                    "Model": res.get("model_name", model_name),
                    "Timestamp": res.get("timestamp", datetime.now().isoformat()),
                })
                progress_bar.progress((idx + 1) / len(texts_list))

            status_text.text("Batch classification complete!")
            st.toast("Batch analysis finished successfully!", icon="🚀")

            res_df = pd.DataFrame(results)
            st.session_state["last_batch_results_df"] = res_df

    if "last_batch_results_df" in st.session_state:
        res_df = st.session_state["last_batch_results_df"]
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Batch Results")

        filter_choice = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"], key="batch_filter_sel")
        display_df = res_df if filter_choice == "All" else res_df[res_df["Predicted Sentiment"] == filter_choice]

        st.dataframe(display_df, use_container_width=True)

        csv_out = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Analyzed CSV",
            data=csv_out,
            file_name=f"sentimentlab_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_dl_batch_csv",
        )