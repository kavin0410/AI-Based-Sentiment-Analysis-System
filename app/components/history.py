"""History Component for AI Sentiment Intelligence.

Provides prediction history view, filtering, searching, summary KPIs, and CSV export.
"""

from datetime import datetime
import pandas as pd
import streamlit as st
import config


def render_history_page():
    """Render the Prediction History page."""
    st.markdown("## 🕘 Prediction History")
    st.caption("Review, filter, search, and export inference queries executed during the current active session.")

    history_mgr = st.session_state.get("history_manager")
    if not history_mgr or not history_mgr.get_history():
        st.info("No predictions recorded in this session yet. Use the **Analyzer** or **Overview** to classify text.")
        return

    history_df = history_mgr.to_dataframe()
    total_preds = len(history_df)

    # -------------------------------------------------------------------------
    # 1. Session Summary KPIs
    # -------------------------------------------------------------------------
    s_counts = history_df["sentiment"].value_counts()
    pos_cnt = s_counts.get("Positive", 0)
    neg_cnt = s_counts.get("Negative", 0)
    neu_cnt = s_counts.get("Neutral", 0)

    # Calculate average confidence for probability models
    prob_rows = history_df[history_df["score_type"] == "probability"]
    avg_conf = (prob_rows["score"].mean() * 100) if not prob_rows.empty else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Predictions", f"{total_preds}")
    with kpi2:
        st.metric("Positive %", f"{(pos_cnt / total_preds * 100):.1f}%", f"{pos_cnt} items")
    with kpi3:
        st.metric("Negative %", f"{(neg_cnt / total_preds * 100):.1f}%", f"{neg_cnt} items")
    with kpi4:
        st.metric("Neutral %", f"{(neu_cnt / total_preds * 100):.1f}%", f"{neu_cnt} items")
    with kpi5:
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Search & Filters
    # -------------------------------------------------------------------------
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        search_query = st.text_input("🔍 Search text in history:", placeholder="Filter by keyword...", key="hist_search_box")
    with col_f2:
        sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Negative", "Neutral"], key="hist_sent_filter")

    # Apply filters
    filtered_df = history_df.copy()
    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter]
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["text"].str.contains(search_query.strip(), case=False, na=False)]

    # -------------------------------------------------------------------------
    # 3. Formatted Table View
    # -------------------------------------------------------------------------
    if filtered_df.empty:
        st.warning("No records matched your search/filter criteria.")
    else:
        st.caption(f"Showing **{len(filtered_df)}** of **{total_preds}** entries (limit: {config.PREDICTION_HISTORY_LIMIT})")

        display_df = filtered_df.copy()
        display_df["Score / Conf"] = display_df.apply(
            lambda r: f"{r['score'] * 100:.1f}%" if r["score_type"] == "probability" else f"{r['score']:.4f} (Decision)",
            axis=1,
        )
        display_df = display_df.rename(columns={
            "timestamp": "Timestamp",
            "text": "Text Input",
            "sentiment": "Sentiment",
            "model": "Model Used",
        })

        st.dataframe(
            display_df[["Timestamp", "Text Input", "Sentiment", "Score / Conf", "Model Used"]].sort_values("Timestamp", ascending=False),
            use_container_width=True,
            height=320,
        )

    # -------------------------------------------------------------------------
    # 4. Actions: Export CSV and Clear
    # -------------------------------------------------------------------------
    col_a1, col_a2 = st.columns([3, 1])
    with col_a1:
        csv_bytes = history_mgr.to_csv()
        st.download_button(
            label="📥 Export Session History (CSV)",
            data=csv_bytes,
            file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_export_hist_page",
        )
    with col_a2:
        if st.button("🗑️ Clear History", use_container_width=True, key="btn_clear_hist_page"):
            history_mgr.clear()
            st.rerun()
