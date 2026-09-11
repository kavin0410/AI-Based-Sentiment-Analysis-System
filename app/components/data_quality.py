"""Data Center Component for AI Sentiment Intelligence.

Provides dataset overview, raw vs cleaned corpus inspection, class distribution metrics,
missing value / duplicate audit, and file download capabilities.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import config
from src.data_loader import (
    calculate_text_statistics,
    get_default_dataset_path,
    load_dataset,
)


def render_data_quality_page():
    """Render the Data Center page."""
    st.markdown("## 🗄️ Data Center")
    st.caption("Centralized data asset management, data quality audits, corpus inspection, and dataset exports.")

    raw_path = get_default_dataset_path()

    try:
        raw_df = load_dataset(raw_path)
    except Exception as e:
        st.error(f"Error loading raw corpus: {e}")
        return

    total_records = len(raw_df)
    pos_cnt = (raw_df[config.SENTIMENT_COLUMN] == "Positive").sum()
    neg_cnt = (raw_df[config.SENTIMENT_COLUMN] == "Negative").sum()
    neu_cnt = (raw_df[config.SENTIMENT_COLUMN] == "Neutral").sum()
    missing_cnt = raw_df.isnull().sum().sum()
    dup_cnt = raw_df.duplicated().sum()

    # -------------------------------------------------------------------------
    # 1. Dataset Health Metrics
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Records", f"{total_records}")
    with col2:
        st.metric("Positive", f"{pos_cnt}")
    with col3:
        st.metric("Negative", f"{neg_cnt}")
    with col4:
        st.metric("Neutral", f"{neu_cnt}")
    with col5:
        st.metric("Missing Rows", f"{missing_cnt}")
    with col6:
        st.metric("Duplicate Rows", f"{dup_cnt}")

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Raw vs Cleaned Dataset Previews
    # -------------------------------------------------------------------------
    col_raw, col_clean = st.columns(2)

    with col_raw:
        st.markdown("##### 📄 Raw Corpus (`data/raw/sample_data.csv`)")
        st.caption("Unmodified input data containing raw strings, URLs, punctuation, and contractions.")
        st.dataframe(raw_df, use_container_width=True, height=280)

    with col_clean:
        st.markdown("##### 🧼 Cleaned Corpus (`data/processed/cleaned_dataset.csv`)")
        if config.CLEANED_DATA_FILE.exists():
            clean_df = pd.read_csv(config.CLEANED_DATA_FILE)
            st.caption("Normalized strings with negation preservation and contractions expanded.")
            st.dataframe(clean_df, use_container_width=True, height=280)
        else:
            clean_df = None
            st.info("Cleaned dataset not found.")

    # -------------------------------------------------------------------------
    # 3. Downloads & Exports
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("### 📥 Dataset Downloads")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if config.CLEANED_DATA_FILE.exists():
            clean_csv_bytes = config.CLEANED_DATA_FILE.read_bytes()
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=clean_csv_bytes,
                file_name="cleaned_sentiment_dataset.csv",
                mime="text/csv",
                key="btn_dl_clean_csv",
            )
        else:
            st.button("Cleaned Dataset Unavailable", disabled=True)

    with col_d2:
        if config.DATA_QUALITY_REPORT_TXT.exists():
            with open(config.DATA_QUALITY_REPORT_TXT, "r", encoding="utf-8") as f:
                report_text = f.read()

            st.download_button(
                label="📥 Download Data Quality Report (.txt)",
                data=report_text,
                file_name="data_quality_report.txt",
                mime="text/plain",
                key="btn_dl_quality_txt",
            )
        else:
            st.button("Quality Report Unavailable", disabled=True)