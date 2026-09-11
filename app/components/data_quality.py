"""Data Quality Component for AI Sentiment Intelligence (Stage 6).

Provides dataset overview, validation checks, statistics, raw vs cleaned dataset previews, and data quality report downloads.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.data_loader import (
    calculate_text_statistics,
    get_default_dataset_path,
    load_dataset,
)


def render_data_quality_page():
    """Render the Data & Quality Audit page."""
    st.markdown("## 🛡️ Dataset Audit & Quality Engineering")
    st.caption("Comprehensive data audit, missing value checks, duplicate inspection, and raw/cleaned corpus previews.")

    raw_path = get_default_dataset_path()
    st.caption(f"Active raw dataset path: `{raw_path.resolve()}`")

    try:
        raw_df = load_dataset(raw_path)
        stats = calculate_text_statistics(raw_df, text_column=config.TEXT_COLUMN)

        # Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Records", len(raw_df))
        with col_m2:
            st.metric("Positive Samples", (raw_df[config.SENTIMENT_COLUMN] == "Positive").sum())
        with col_m3:
            st.metric("Negative Samples", (raw_df[config.SENTIMENT_COLUMN] == "Negative").sum())
        with col_m4:
            st.metric("Neutral Samples", (raw_df[config.SENTIMENT_COLUMN] == "Neutral").sum())

        st.divider()

        # Quality Details & Data Tables
        col_tbl1, col_tbl2 = st.columns(2)
        with col_tbl1:
            st.markdown("##### 📄 Raw Dataset Preview (`data/raw/sample_data.csv`)")
            st.dataframe(raw_df.head(10), use_container_width=True)

        with col_tbl2:
            st.markdown("##### 🧼 Cleaned Dataset Preview (`data/processed/cleaned_dataset.csv`)")
            if config.CLEANED_DATA_FILE.exists():
                clean_df = pd.read_csv(config.CLEANED_DATA_FILE)
                st.dataframe(clean_df.head(10), use_container_width=True)
            else:
                st.info("Run `python run_stage2.py` to generate the cleaned dataset.")

        # Data Quality Audit Text Report View & Download
        if config.DATA_QUALITY_REPORT_TXT.exists():
            st.divider()
            st.markdown("##### 📋 Data Quality Audit Report")
            with open(config.DATA_QUALITY_REPORT_TXT, "r", encoding="utf-8") as f:
                report_txt = f.read()

            with st.expander("View Full Text Quality Report", expanded=True):
                st.code(report_txt, language="text")

            st.download_button(
                label="📥 Download Data Quality Report (.txt)",
                data=report_txt,
                file_name="data_quality_report.txt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
