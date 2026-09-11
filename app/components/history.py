"""History Component for SentimentAI (SentimentLab).

Provides prediction history:
- Timestamp, text preview, sentiment, confidence, model
- Search and sentiment filtering
- CSV export
"""

from datetime import datetime
import pandas as pd
import streamlit as st
import config


def render_history_page():
    """Render the History log page."""
    st.markdown("## ◷ History")
    st.caption("Inspect, filter, search, and export past sentiment classifications from the active session.")

    history_mgr = st.session_state.get("history_manager")
    if not history_mgr or not history_mgr.get_history():
        st.info("No prediction queries logged yet. Use **✦ Analyze** to classify text.")
        return

    history_df = history_mgr.to_dataframe()
    total_preds = len(history_df)

    # -------------------------------------------------------------------------
    # 1. Search & Filter Bar
    # -------------------------------------------------------------------------
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        search_kw = st.text_input("Search query text:", placeholder="Filter by word or phrase...", key="hist_search_input")
    with col_f2:
        filter_sent = st.selectbox("Sentiment Filter:", ["All", "Positive", "Neutral", "Negative"], key="hist_filter_sent")

    filtered_df = history_df.copy()
    if filter_sent != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == filter_sent]
    if search_kw.strip():
        filtered_df = filtered_df[filtered_df["text"].str.contains(search_kw.strip(), case=False, na=False)]

    # -------------------------------------------------------------------------
    # 2. Formatted Log Table
    # -------------------------------------------------------------------------
    st.caption(f"Displaying **{len(filtered_df)}** of **{total_preds}** entries")

    if filtered_df.empty:
        st.warning("No records matched your search query.")
    else:
        disp_df = filtered_df.copy()
        disp_df["Confidence"] = disp_df.apply(
            lambda r: f"{r['score'] * 100:.1f}%" if r["score_type"] == "probability" else f"{r['score']:.4f} (Score)",
            axis=1,
        )
        disp_df["Text Preview"] = disp_df["text"].apply(lambda t: t[:75] + "..." if len(t) > 75 else t)
        disp_df = disp_df.rename(columns={
            "timestamp": "Timestamp",
            "sentiment": "Sentiment",
            "model": "Model",
        })

        st.dataframe(
            disp_df[["Timestamp", "Text Preview", "Sentiment", "Confidence", "Model"]].sort_values("Timestamp", ascending=False),
            use_container_width=True,
            height=340,
        )

    # -------------------------------------------------------------------------
    # 3. Actions
    # -------------------------------------------------------------------------
    col_act1, col_act2 = st.columns([3, 1])
    with col_act1:
        csv_bytes = history_mgr.to_csv()
        st.download_button(
            label="Download Log (CSV)",
            data=csv_bytes,
            file_name=f"sentiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_dl_hist",
        )
    with col_act2:
        if st.button("Clear Log", use_container_width=True, key="btn_clear_hist"):
            history_mgr.clear()
            st.rerun()