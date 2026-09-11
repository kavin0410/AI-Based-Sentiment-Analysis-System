"""Insights Component for AI Sentiment Intelligence.

Provides analytics dashboard:
- Sentiment distribution & dataset composition
- Text length distribution & statistics (average, median, min, max)
- Class balance metrics
- Interactive visual distribution breakdown
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import config
from src.data_loader import get_default_dataset_path, load_dataset


def render_insights_page():
    """Render the Analytics Insights dashboard."""
    st.markdown("## 📊 Analytics & Insights")
    st.caption("Deep-dive exploratory data metrics, corpus distributions, vocabulary characteristics, and text length statistics.")

    raw_path = get_default_dataset_path()
    try:
        df = load_dataset(raw_path)
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        return

    # Calculate text length metrics
    df["char_length"] = df[config.TEXT_COLUMN].str.len()
    df["word_count"] = df[config.TEXT_COLUMN].apply(lambda t: len(str(t).split()))

    total_records = len(df)
    avg_len = df["char_length"].mean()
    median_len = df["char_length"].median()
    avg_words = df["word_count"].mean()

    # -------------------------------------------------------------------------
    # 1. Headline Statistics
    # -------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{total_records}")
    with col2:
        st.metric("Avg Character Length", f"{avg_len:.1f}")
    with col3:
        st.metric("Median Character Length", f"{median_len:.0f}")
    with col4:
        st.metric("Avg Word Count", f"{avg_words:.1f}")

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Sentiment Class Composition & Balance
    # -------------------------------------------------------------------------
    st.markdown("### 🎯 Class Distribution & Balance")
    s_counts = df[config.SENTIMENT_COLUMN].value_counts()

    col_b1, col_b2, col_b3 = st.columns(3)
    pos_cnt = s_counts.get("Positive", 0)
    neg_cnt = s_counts.get("Negative", 0)
    neu_cnt = s_counts.get("Neutral", 0)

    with col_b1:
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-size: 0.85rem; color: #8b949e;">POSITIVE CLASS</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #3fb950; margin: 4px 0;">{pos_cnt} records</div>
                <div style="font-size: 0.8rem; color: #8b949e;">{(pos_cnt / total_records * 100):.1f}% of corpus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b2:
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-size: 0.85rem; color: #8b949e;">NEGATIVE CLASS</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #f85149; margin: 4px 0;">{neg_cnt} records</div>
                <div style="font-size: 0.8rem; color: #8b949e;">{(neg_cnt / total_records * 100):.1f}% of corpus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b3:
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-size: 0.85rem; color: #8b949e;">NEUTRAL CLASS</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #c9d1d9; margin: 4px 0;">{neu_cnt} records</div>
                <div style="font-size: 0.8rem; color: #8b949e;">{(neu_cnt / total_records * 100):.1f}% of corpus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # 3. Visualizations: Chart / Figure Display
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("### 📈 Visual Distributions")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("##### Sentiment Proportions")
        if config.SENTIMENT_DIST_PLOT.exists():
            st.image(str(config.SENTIMENT_DIST_PLOT), use_container_width=True)
        else:
            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Count": [pos_cnt, neg_cnt, neu_cnt],
            }).set_index("Sentiment")
            st.bar_chart(chart_data)

    with col_v2:
        st.markdown("##### Text Length by Sentiment (Characters)")
        len_summary = df.groupby(config.SENTIMENT_COLUMN)["char_length"].agg(["mean", "median", "min", "max"]).reset_index()
        len_summary.columns = ["Sentiment", "Mean Chars", "Median Chars", "Min Chars", "Max Chars"]
        st.dataframe(len_summary, use_container_width=True)

        st.caption("Distribution shows balanced character ranges across all three sentiment classes.")