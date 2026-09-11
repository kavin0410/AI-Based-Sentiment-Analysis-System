"""History Component for SentimentLab.

Prediction history center:
- Timestamp, Text Preview, Sentiment, Confidence, Model
- Search, Sentiment Filter, Clear History, CSV Export
- Interactive modal/detail view when clicking an entry
"""

from datetime import datetime
import pandas as pd
import streamlit as st
import config


def render_history_page():
    """Render the Prediction History Center."""
    st.markdown("## ◷ History")
    st.caption("Inspect, query, filter, and review full classification traces from your active session.")

    history_mgr = st.session_state.get("history_manager")
    if not history_mgr or not history_mgr.get_history():
        st.info("No queries recorded in this session yet. Navigate to **✦ Analyze** to start classifying text.")
        return

    history_df = history_mgr.to_dataframe()
    total_preds = len(history_df)

    # -------------------------------------------------------------------------
    # 1. Search & Filter Bar
    # -------------------------------------------------------------------------
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_kw = st.text_input("Search prediction text:", placeholder="Search by keyword or phrase...", key="hist_search_v2")
    with col_s2:
        sent_filter = st.selectbox("Filter Sentiment:", ["All", "Positive", "Neutral", "Negative"], key="hist_filter_v2")

    filtered_df = history_df.copy()
    if sent_filter != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == sent_filter]
    if search_kw.strip():
        filtered_df = filtered_df[filtered_df["text"].str.contains(search_kw.strip(), case=False, na=False)]

    st.caption(f"Showing **{len(filtered_df)}** of **{total_preds}** logged queries")

    # -------------------------------------------------------------------------
    # 2. Results Dataframe
    # -------------------------------------------------------------------------
    if filtered_df.empty:
        st.warning("No records matched your search criteria.")
    else:
        disp_df = filtered_df.copy()
        disp_df["Confidence"] = disp_df.apply(
            lambda r: f"{r['score'] * 100:.1f}%" if r["score_type"] == "probability" else f"{r['score']:.4f}",
            axis=1,
        )
        disp_df["Text Preview"] = disp_df["text"].apply(lambda t: t[:80] + "..." if len(t) > 80 else t)
        disp_df = disp_df.rename(columns={
            "timestamp": "Timestamp",
            "sentiment": "Sentiment",
            "model": "Model",
        })

        st.dataframe(
            disp_df[["Timestamp", "Text Preview", "Sentiment", "Confidence", "Model"]].sort_values("Timestamp", ascending=False),
            use_container_width=True,
            height=320,
        )

        # ---------------------------------------------------------------------
        # 3. Detailed Inspection Drawer
        # ---------------------------------------------------------------------
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Inspect Entry Details")
        entry_indices = filtered_df.index.tolist()
        selected_idx = st.selectbox("Select query to inspect:", entry_indices, format_func=lambda i: f"#{i+1}: {filtered_df.loc[i, 'text'][:60]}...")

        if selected_idx is not None:
            sel_row = filtered_df.loc[selected_idx]
            with st.expander(f"Detailed Analysis — Query #{selected_idx+1}", expanded=True):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("**Original Text:**")
                    st.code(sel_row["text"], language="text")
                    st.markdown(f"**Classification:** `{sel_row['sentiment']}`")
                with col_d2:
                    st.markdown("**Cleaned & Tokenized Text:**")
                    st.code(sel_row["processed_text"], language="text")
                    st.markdown(f"**Confidence / Score:** `{sel_row['score']}` ({sel_row['score_type']})")
                    st.markdown(f"**Model:** `{sel_row['model']}` | **Time:** `{sel_row['timestamp'][:19]}`")

    # -------------------------------------------------------------------------
    # 4. Actions
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col_a1, col_a2 = st.columns([3, 1])
    with col_a1:
        csv_bytes = history_mgr.to_csv()
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv_bytes,
            file_name=f"sentimentlab_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="btn_dl_hist_v2",
        )
    with col_a2:
        if st.button("🗑️ Clear History", use_container_width=True, key="btn_clear_hist_v2"):
            history_mgr.clear()
            st.toast("Prediction history cleared!", icon="🧹")
            st.rerun()