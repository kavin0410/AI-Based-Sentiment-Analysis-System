"""Data Component for SentimentLab — Premium Light UI.

Dataset center:
- Dataset overview metrics
- Raw vs Cleaned datasets
- Quality metrics
- Downloads
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
    """Render the Data center page."""
    st.markdown(
        """
        <div class="sl-page-header">
            <div class="sl-page-eyebrow">DATA MANAGEMENT</div>
            <h1 class="sl-page-title">Data</h1>
            <p class="sl-page-subtitle">
                Corpus validation, missing value audits, duplicate detection, and dataset exports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_path = get_default_dataset_path()
    try:
        raw_df = load_dataset(raw_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    tot_records = len(raw_df)
    pos_cnt = (raw_df[config.SENTIMENT_COLUMN] == "Positive").sum()
    neg_cnt = (raw_df[config.SENTIMENT_COLUMN] == "Negative").sum()
    neu_cnt = (raw_df[config.SENTIMENT_COLUMN] == "Neutral").sum()
    missing_cnt = raw_df.isnull().sum().sum()
    dup_cnt = raw_df.duplicated().sum()

    # ------------------------------------------------------------------
    # 1. Dataset Metrics
    # ------------------------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("Total Records", f"{tot_records}")
    with c2: st.metric("Positive", f"{pos_cnt}")
    with c3: st.metric("Neutral", f"{neu_cnt}")
    with c4: st.metric("Negative", f"{neg_cnt}")
    with c5: st.metric("Missing Values", f"{missing_cnt}")
    with c6: st.metric("Duplicates", f"{dup_cnt}")

    st.markdown("<div class='sl-spacer-md'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2. Raw vs Cleaned Datasets
    # ------------------------------------------------------------------
    col_raw, col_clean = st.columns(2, gap="medium")

    with col_raw:
        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Raw Dataset</div>'
            '<div class="sl-text-muted" style="margin-bottom:0.75rem;">Unprocessed text with punctuation, numbers, and contractions.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(raw_df, use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_clean:
        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Cleaned Dataset</div>'
            '<div class="sl-text-muted" style="margin-bottom:0.75rem;">Normalized corpus with expanded contractions and preserved negations.</div>',
            unsafe_allow_html=True,
        )
        if config.CLEANED_DATA_FILE.exists():
            clean_df = pd.read_csv(config.CLEANED_DATA_FILE)
            st.dataframe(clean_df, use_container_width=True, height=300)
        else:
            st.info("Cleaned dataset not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sl-spacer-md'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. Downloads
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="sl-section-label">Export Datasets</div>',
        unsafe_allow_html=True,
    )
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if config.CLEANED_DATA_FILE.exists():
            st.download_button(
                label="Download Cleaned Dataset (CSV)",
                data=config.CLEANED_DATA_FILE.read_bytes(),
                file_name="cleaned_sentiment_dataset.csv",
                mime="text/csv",
                key="btn_dl_cleaned",
            )
        else:
            st.button("Cleaned Dataset Unavailable", disabled=True)

    with col_d2:
        if config.DATA_QUALITY_REPORT_TXT.exists():
            with open(config.DATA_QUALITY_REPORT_TXT, "r", encoding="utf-8") as f:
                rep_txt = f.read()
            st.download_button(
                label="Download Quality Audit (.txt)",
                data=rep_txt,
                file_name="data_quality_report.txt",
                mime="text/plain",
                key="btn_dl_quality",
            )
        else:
            st.button("Quality Audit Unavailable", disabled=True)