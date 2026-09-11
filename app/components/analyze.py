"""Analyze Component for SentimentLab — Premium Light UI.

Primary product experience:
1. Single Text Analysis with result card, confidence bar, pipeline explainability
2. Text Comparison — two-panel side-by-side
3. Batch CSV Analysis — polished upload & batch run
"""

from datetime import datetime
import pandas as pd
import streamlit as st
import config
from src.model_loader import load_best_model, load_vectorizer, validate_model_artifacts
from src.predict import predict_batch, predict_sentiment
from src.preprocessing import expand_contractions, lemmatize_tokens, remove_stopwords, tokenize_text
from src.data_cleaning import clean_text_basic

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_analyze_page():
    """Render the Analyze workspace."""
    st.markdown(
        """
        <div class="sl-page-header">
            <div class="sl-page-eyebrow">ANALYSIS WORKSPACE</div>
            <h1 class="sl-page-title">Analyze</h1>
            <p class="sl-page-subtitle">
                Real-time sentiment classification, side-by-side text comparison,
                and large-batch CSV inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    val_report = validate_model_artifacts()
    if not val_report["valid"]:
        st.error("Model artifacts unavailable. Ensure models are trained.")
        return

    try:
        model, model_name, meta = load_best_model()
        vectorizer = load_vectorizer()
    except Exception as e:
        st.error(f"Error loading inference engine: {e}")
        return

    tab1, tab2, tab3 = st.tabs(["Single Text", "Compare Texts", "Batch CSV"])

    with tab1:
        _render_single(model, model_name, vectorizer)
    with tab2:
        _render_comparison(model, model_name, vectorizer)
    with tab3:
        _render_batch(model, model_name, vectorizer)


def _set_preset_callback(preset_text: str):
    st.session_state["analyze_input_text"] = preset_text


def _clear_callback():
    st.session_state["analyze_input_text"] = ""
    st.session_state.pop("last_single_result", None)


def _render_single(model, model_name, vectorizer):
    """Single text analysis panel."""
    st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)

    # Initialize default input text
    if "analyze_input_text" not in st.session_state:
        st.session_state["analyze_input_text"] = "The build quality is extraordinary, setup took less than two minutes, and customer support was outstanding!"

    presets = {
        "Positive": "The build quality is extraordinary, setup took less than two minutes, and customer support was outstanding!",
        "Negative": "Terrible customer service. The device arrived damaged and the company refused to issue a refund.",
        "Neutral":  "The shipment arrived on Wednesday. It contains three charging cables and instructions.",
        "Mixed":    "The screen display is breathtakingly vibrant, but the battery drains in under three hours.",
        "Negation": "I didn't think the performance would be bad, but it isn't quite as fast as advertised.",
    }

    st.markdown('<div class="sl-text-label">Quick Presets</div>', unsafe_allow_html=True)
    p_cols = st.columns(5)
    for i, (p_name, p_text) in enumerate(presets.items()):
        with p_cols[i]:
            st.button(
                p_name,
                key=f"preset_btn_{p_name}",
                use_container_width=True,
                on_click=_set_preset_callback,
                args=(p_text,),
            )

    st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)

    # Input card
    st.markdown('<div class="sl-card">', unsafe_allow_html=True)
    input_text = st.text_area(
        "Input Text",
        key="analyze_input_text",
        height=140,
        max_chars=config.MAX_INPUT_LENGTH,
        placeholder="Paste a review, comment, feedback, message, or any text...",
        label_visibility="collapsed",
    )
    char_cnt = len(st.session_state.get("analyze_input_text", ""))
    st.caption(f"{char_cnt} / {config.MAX_INPUT_LENGTH} characters")

    col_run, col_clear, _ = st.columns([3, 1, 3])
    with col_run:
        btn_analyze = st.button("Analyze Sentiment \u2192", type="primary",
                                use_container_width=True, key="btn_single_run")
    with col_clear:
        st.button("Clear", key="btn_single_clear", use_container_width=True, on_click=_clear_callback)
    st.markdown("</div>", unsafe_allow_html=True)

    # Execute Prediction
    if btn_analyze:
        raw_to_analyze = st.session_state.get("analyze_input_text", "").strip()
        if not raw_to_analyze:
            st.warning("Please enter text to analyze.")
        else:
            with st.spinner("Processing text through NLP pipeline..."):
                result = predict_sentiment(raw_to_analyze, model=model,
                                          vectorizer=vectorizer, model_name=model_name)
                st.session_state["last_single_result"] = result
                hm = st.session_state.get("history_manager")
                if hm and result.get("valid"):
                    hm.add_prediction(result)
                st.toast("Analysis complete!", icon="\u2705")

    # Display Results
    result = st.session_state.get("last_single_result")
    if result and result.get("valid"):
        st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)
        sentiment = result["sentiment"]
        score     = result["score"]
        score_type= result["score_type"]
        m_used    = result.get("model_name", model_name)
        timestamp = result.get("timestamp", datetime.now().isoformat())
        probs     = result.get("probabilities", {})
        pos_p = probs.get("Positive", 0.0)
        neu_p = probs.get("Neutral",  0.0)
        neg_p = probs.get("Negative", 0.0)

        badge_map = {
            "Positive": "sl-badge-positive",
            "Negative": "sl-badge-negative",
            "Neutral":  "sl-badge-neutral",
        }
        badge_cls = badge_map.get(sentiment, "sl-badge-neutral")

        if sentiment == "Positive":
            marker_pct = 50 + int(score * 45)
        elif sentiment == "Negative":
            marker_pct = 50 - int(score * 45)
        else:
            marker_pct = 50

        col_r1, col_r2 = st.columns([1, 1], gap="medium")

        with col_r1:
            st.markdown(
                f"""
                <div class="sl-card">
                    <div class="sl-text-label">Sentiment Classification</div>
                    <div style="margin: 0.75rem 0 1.25rem 0;">
                        <span class="sl-badge {badge_cls} sl-badge-lg">{sentiment}</span>
                    </div>
                    <div style="font-size:0.88rem; color:#475569; line-height:2;">
                        <span style="color:#64748B;">Confidence</span>&nbsp;
                        <strong style="color:#0F172A; font-size:1.4rem; font-weight:800;">
                            {score*100:.1f}%
                        </strong><br/>
                        <span style="color:#64748B;">Model Used</span>&nbsp;
                        <strong style="color:#6366F1; font-weight:700;">{m_used}</strong><br/>
                        <span style="color:#64748B;">Timestamp</span>&nbsp;
                        <span style="color:#1E293B; font-size:0.82rem; font-weight:500;">
                            {timestamp[:19].replace('T', ' ')}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r2:
            st.markdown(
                '<div class="sl-card"><div class="sl-text-label">Probability Distribution</div>',
                unsafe_allow_html=True,
            )
            if score_type == "probability" and HAS_PLOTLY:
                cats   = ["Positive", "Neutral", "Negative"]
                values = [pos_p, neu_p, neg_p]
                colors = ["#10B981", "#6366F1", "#EF4444"]
                fig_h = go.Figure(go.Bar(
                    x=values, y=cats, orientation="h",
                    marker_color=colors,
                    text=[f"{v*100:.1f}%" for v in values],
                    textposition="outside",
                    textfont=dict(family="Plus Jakarta Sans", size=12, color="#0F172A"),
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                ))
                fig_h.update_layout(
                    xaxis=dict(range=[0, 1.15], tickformat=".0%",
                               tickfont=dict(family="Plus Jakarta Sans", size=11, color="#64748B"),
                               gridcolor="rgba(99,102,241,0.08)"),
                    yaxis=dict(tickfont=dict(family="Plus Jakarta Sans", size=13, color="#1E293B")),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=60),
                    height=180, showlegend=False,
                )
                st.plotly_chart(fig_h, use_container_width=True,
                                config={"displayModeBar": False})
            elif score_type == "probability":
                for cls, p_val in [("Positive", pos_p), ("Neutral", neu_p), ("Negative", neg_p)]:
                    st.write(f"**{cls}:** {p_val*100:.1f}%")
                    st.progress(min(max(p_val, 0.0), 1.0))
            else:
                for cls in ["Negative", "Neutral", "Positive"]:
                    st.write(f"**{cls}:** `{probs.get(cls, 0.0):.4f}` (Decision Score)")
            st.markdown("</div>", unsafe_allow_html=True)

        # Sentiment Meter
        st.markdown(
            f"""
            <div class="sl-meter">
                <div class="sl-meter-labels">
                    <span style="color:#B91C1C;">Negative</span>
                    <span style="color:#4338CA;">Neutral</span>
                    <span style="color:#047857;">Positive</span>
                </div>
                <div class="sl-meter-track">
                    <div class="sl-meter-marker" style="left:{marker_pct}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Explainability panel
        st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)
        with st.expander("How SentimentLab reached this result (NLP Trace)", expanded=False):
            raw_t    = result.get("text", st.session_state.get("analyze_input_text", ""))
            s1_exp   = expand_contractions(raw_t)
            s2_clean = clean_text_basic(s1_exp)
            s3_tok   = tokenize_text(s2_clean)
            s4_stop  = remove_stopwords(s3_tok)
            s5_lem   = lemmatize_tokens(s4_stop)
            s6_final = result.get("processed_text", " ".join(s5_lem))

            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.markdown("**1. Original Text**")
                st.code(raw_t, language="text")
                st.markdown("**2. Cleaned & Normalized**")
                st.code(s2_clean, language="text")
                st.markdown("**3. Tokens**")
                st.write(s3_tok)
            with c_e2:
                st.markdown("**4. Stopword Filtering (Negations Retained)**")
                st.write(s4_stop)
                st.markdown("**5. Lemmatized Tokens**")
                st.write(s5_lem)
                st.markdown("**6. Final Vectorizer Input**")
                st.code(s6_final, language="text")
            st.caption(f"Vectorized into {result.get('vocab_size', 570)} TF-IDF features \u00b7 classified by {m_used}.")


def _render_comparison(model, model_name, vectorizer):
    """Side-by-side text comparison."""
    st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sl-card" style="margin-bottom:1rem;">'
        '<div class="sl-section-title">Text Comparison</div>'
        '<div class="sl-text-muted" style="margin-bottom:1rem;">'
        'Compare sentiment across two independent texts side-by-side.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown('<div class="sl-text-label">Text A</div>', unsafe_allow_html=True)
        text_a = st.text_area(
            "Text A",
            value="The camera quality on this new phone is breathtaking and works seamlessly.",
            height=120, key="cmp_a", label_visibility="collapsed",
        )
    with col_b:
        st.markdown('<div class="sl-text-label">Text B</div>', unsafe_allow_html=True)
        text_b = st.text_area(
            "Text B",
            value="The battery life is frustratingly short and overheats after 20 minutes.",
            height=120, key="cmp_b", label_visibility="collapsed",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Compare Texts \u2192", type="primary", use_container_width=True, key="btn_compare"):
        if not text_a.strip() or not text_b.strip():
            st.warning("Please enter text in both panels.")
            return
        with st.spinner("Analyzing both texts..."):
            res_a = predict_sentiment(text_a, model=model, vectorizer=vectorizer, model_name=model_name)
            res_b = predict_sentiment(text_b, model=model, vectorizer=vectorizer, model_name=model_name)

        col_ra, col_rb = st.columns(2, gap="medium")
        badge_map = {"Positive": "sl-badge-positive", "Negative": "sl-badge-negative", "Neutral": "sl-badge-neutral"}
        for tcol, res, lbl in [(col_ra, res_a, "Text A"), (col_rb, res_b, "Text B")]:
            with tcol:
                sent  = res["sentiment"]
                score = res["score"]
                probs = res.get("probabilities", {})
                st.markdown(
                    f"""
                    <div class="sl-card">
                        <div class="sl-text-label">{lbl} Result</div>
                        <div style="margin:0.6rem 0;">
                            <span class="sl-badge {badge_map.get(sent,'sl-badge-neutral')} sl-badge-lg">{sent}</span>
                        </div>
                        <div style="font-size:0.85rem;color:#475569;">
                            Confidence: <strong style="color:#0F172A;font-size:1.2rem;">{score*100:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if res["score_type"] == "probability":
                    for cls in ["Positive", "Neutral", "Negative"]:
                        st.write(f"**{cls}:** {probs.get(cls,0.0)*100:.1f}%")
                        st.progress(min(max(probs.get(cls, 0.0), 0.0), 1.0))

        st.info(
            f"**Comparison:** Text A \u2192 **{res_a['sentiment']}** ({res_a['score']*100:.1f}%) "
            f"| Text B \u2192 **{res_b['sentiment']}** ({res_b['score']*100:.1f}%)"
        )


def _render_batch(model, model_name, vectorizer):
    """Batch CSV upload and analysis."""
    st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sl-card">'
        '<div class="sl-section-title">Batch CSV Analysis</div>'
        '<div class="sl-text-muted" style="margin-bottom:1rem;">'
        'Upload a CSV file containing a <code>text</code> column to run batch classification.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_uploader",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
            return

        st.success(f"Uploaded successfully \u2014 **{len(df)} records** found.")
        st.table(df.head(5))

        if st.button("Run Batch Prediction \u2192", type="primary", key="btn_batch_run"):
            texts_list = df["text"].dropna().astype(str).tolist()
            progress_bar = st.progress(0.0)
            status_text  = st.empty()
            status_text.text(f"Classifying {len(texts_list)} texts...")

            results = []
            for idx, txt in enumerate(texts_list):
                res = predict_sentiment(txt, model=model, vectorizer=vectorizer, model_name=model_name)
                results.append({
                    "Text": txt,
                    "Predicted Sentiment": res.get("sentiment", "Unknown"),
                    "Confidence": (
                        f"{res.get('score', 0.0)*100:.1f}%"
                        if res.get("score_type") == "probability"
                        else f"{res.get('score', 0.0):.4f}"
                    ),
                    "Model": res.get("model_name", model_name),
                    "Timestamp": res.get("timestamp", datetime.now().isoformat()),
                })
                progress_bar.progress((idx + 1) / len(texts_list))

            status_text.text("Batch classification complete!")
            st.toast("Batch analysis finished!", icon="\U0001f680")
            st.session_state["last_batch_df"] = pd.DataFrame(results)

    if "last_batch_df" in st.session_state:
        res_df = st.session_state["last_batch_df"]

        st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)
        sc = res_df["Predicted Sentiment"].value_counts()
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("Records", len(res_df))
        with m2: st.metric("Positive", int(sc.get("Positive", 0)))
        with m3: st.metric("Neutral",  int(sc.get("Neutral",  0)))
        with m4: st.metric("Negative", int(sc.get("Negative", 0)))
        with m5:
            try:
                avg_c = res_df["Confidence"].str.rstrip("%").astype(float).mean()
                st.metric("Avg Confidence", f"{avg_c:.1f}%")
            except Exception:
                st.metric("Avg Confidence", "\u2014")

        filt = st.selectbox("Filter:", ["All", "Positive", "Neutral", "Negative"], key="batch_filter")
        disp = res_df if filt == "All" else res_df[res_df["Predicted Sentiment"] == filt]
        st.table(disp.head(25))

        st.download_button(
            label="Download Results (CSV)",
            data=res_df.to_csv(index=False).encode("utf-8"),
            file_name=f"sentimentlab_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_batch_dl",
        )