"""History Component for SentimentLab — Premium Light UI.

Prediction history center:
- Search, Sentiment Filter, Clear History, CSV Export
- Styled table with sentiment pills
- Detailed entry inspection
- 1-click sample query loader
"""

from datetime import datetime
import pandas as pd
import streamlit as st
import config
from src.model_loader import load_best_model, load_vectorizer
from src.predict import predict_sentiment


def _seed_samples(history_mgr):
    """Seed benchmark queries into history manager."""
    try:
        m, mn, _ = load_best_model()
        v = load_vectorizer()
        sample_corpus = [
            "The product quality is exceptional and exceeded all my expectations.",
            "Customer support was quick, friendly, and resolved my issue in minutes.",
            "Terrible customer service. The device arrived damaged and the company refused a refund.",
            "The delivery arrived ahead of schedule and the packaging was in perfect condition.",
            "The shipment arrived on Wednesday with two cables and instructions.",
            "I didn't think the performance would be bad, but it isn't quite as fast as advertised.",
            "Great battery life, fantastic display, and very lightweight to carry!",
            "I am neither satisfied nor dissatisfied; it performs exactly as an average device.",
        ]
        for s in sample_corpus:
            res = predict_sentiment(s, model=m, vectorizer=v, model_name=mn)
            if res.get("valid"):
                history_mgr.add_prediction(res)
    except Exception:
        pass


def render_history_page():
    """Render the Prediction History Center."""
    st.markdown(
        """
        <div class="sl-page-header">
            <div class="sl-page-eyebrow">PREDICTION ARCHIVE</div>
            <h1 class="sl-page-title">History</h1>
            <p class="sl-page-subtitle">
                Inspect, filter, and review classification traces from your active session.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_mgr = st.session_state.get("history_manager")
    if not history_mgr:
        from src.predict import PredictionHistoryManager
        history_mgr = PredictionHistoryManager(max_limit=config.PREDICTION_HISTORY_LIMIT)
        st.session_state["history_manager"] = history_mgr

    # If empty, auto-seed or show options
    if not history_mgr.get_history():
        st.markdown(
            """
            <div class="sl-card" style="text-align:center;padding:2.5rem 1.5rem;">
                <div style="font-size:1.3rem;font-weight:700;color:#0F172A;margin-bottom:0.5rem;">
                    No predictions logged yet
                </div>
                <div class="sl-text-muted" style="margin-bottom:1.25rem;">
                    Run custom queries in <strong>Analyze</strong> or populate the table with benchmark test cases.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load Benchmark Queries \u2192", type="primary", key="btn_load_samples"):
            _seed_samples(history_mgr)
            st.toast("Loaded 8 benchmark predictions!", icon="\U0001f4e5")
            st.rerun()
        return

    history_df = history_mgr.to_dataframe()
    total_preds = len(history_df)

    # ------------------------------------------------------------------
    # Search & Filter
    # ------------------------------------------------------------------
    st.markdown('<div class="sl-card">', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_kw = st.text_input("Search", placeholder="Search by keyword or phrase...",
                                  key="hist_search", label_visibility="collapsed")
    with col_s2:
        sent_filter = st.selectbox("Filter", ["All", "Positive", "Neutral", "Negative"],
                                   key="hist_filter", label_visibility="collapsed")

    filtered_df = history_df.copy()
    if sent_filter != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == sent_filter]
    if search_kw.strip():
        filtered_df = filtered_df[
            filtered_df["text"].str.contains(search_kw.strip(), case=False, na=False)
        ]

    st.caption(f"Showing **{len(filtered_df)}** of **{total_preds}** logged queries")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Results Table
    # ------------------------------------------------------------------
    if filtered_df.empty:
        st.warning("No records matched your search criteria.")
    else:
        disp_df = filtered_df.copy()
        disp_df["Confidence"] = disp_df.apply(
            lambda r: f"{r['score'] * 100:.1f}%" if r["score_type"] == "probability" else f"{r['score']:.4f}",
            axis=1,
        )
        disp_df["Text Preview"] = disp_df["text"].apply(
            lambda t: t[:85] + "..." if len(t) > 85 else t
        )
        disp_df = disp_df.rename(columns={
            "timestamp": "Timestamp",
            "sentiment": "Sentiment",
            "model": "Model",
        })

        st.dataframe(
            disp_df[["Timestamp", "Text Preview", "Sentiment", "Confidence", "Model"]].sort_values(
                "Timestamp", ascending=False
            ),
            use_container_width=True,
            height=320,
        )

        # Detailed inspection
        st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Inspect Entry Details</div>',
            unsafe_allow_html=True,
        )
        entry_indices = filtered_df.index.tolist()
        selected_idx = st.selectbox(
            "Select query:",
            entry_indices,
            format_func=lambda i: f"#{i+1}: {filtered_df.loc[i, 'text'][:60]}...",
            key="hist_inspect",
        )

        if selected_idx is not None:
            sel_row = filtered_df.loc[selected_idx]
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Original Text:**")
                st.code(sel_row["text"], language="text")
                badge_map = {"Positive": "sl-badge-positive", "Negative": "sl-badge-negative",
                             "Neutral": "sl-badge-neutral"}
                sent = sel_row["sentiment"]
                st.markdown(
                    f'<span class="sl-badge {badge_map.get(sent, "sl-badge-neutral")} sl-badge-lg">{sent}</span>',
                    unsafe_allow_html=True,
                )
            with col_d2:
                st.markdown("**Processed Text (Vectorized Input):**")
                st.code(sel_row["processed_text"], language="text")
                st.markdown(f"**Confidence:** `{sel_row['score']}` ({sel_row['score_type']})")
                st.markdown(f"**Model:** `{sel_row['model']}` | **Time:** `{str(sel_row['timestamp'])[:19]}`")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    st.markdown("<div class='sl-spacer-sm'></div>", unsafe_allow_html=True)
    col_a1, col_a2, col_a3 = st.columns([2, 1, 1])
    with col_a1:
        csv_bytes = history_mgr.to_csv()
        st.download_button(
            label="Export History (CSV)",
            data=csv_bytes,
            file_name=f"sentimentlab_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_dl_hist",
            use_container_width=True,
        )
    with col_a2:
        if st.button("Reload Samples", use_container_width=True, key="btn_reload_samples"):
            _seed_samples(history_mgr)
            st.toast("Reloaded sample benchmark queries!", icon="\U0001f4e5")
            st.rerun()
    with col_a3:
        if st.button("Clear History", use_container_width=True, key="btn_clear_hist"):
            history_mgr.clear()
            st.toast("History cleared!", icon="\U0001f9f9")
            st.rerun()