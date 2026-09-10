"""Streamlit Web Application for AI Sentiment Intelligence.

Interactive web dashboard for exploring data engineering, NLP preprocessing,
exploratory data analysis, and live text normalization (Stages 1 & 2).
Model inference and classification will be unlocked in Stages 3-6.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.data_cleaning import clean_text_basic
from src.data_loader import (
    calculate_text_statistics,
    get_default_dataset_path,
    load_dataset,
    validate_dataset,
)
from src.preprocessing import (
    expand_contractions,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    tokenize_text,
)


def init_page():
    """Initialize Streamlit page settings and visual branding."""
    st.set_page_config(
        page_title="AI Sentiment Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar():
    """Render sidebar branding and stage progress tracker."""
    with st.sidebar:
        st.markdown("## 🧠 AI Sentiment Intelligence")
        st.caption("AI-Based Sentiment Analysis System Using Machine Learning")
        st.divider()

        st.markdown("### 📌 Project Status")
        st.success("✅ **Stage 1:** Project Architecture & Setup")
        st.success("✅ **Stage 2:** Dataset & NLP Engine")
        st.warning("⏳ **Stage 3:** Feature Extraction & ML Models")
        st.info("🔒 **Stage 4:** Evaluation & Confusion Matrix")
        st.info("🔒 **Stage 5:** Real-time Prediction Engine")
        st.info("🔒 **Stage 6:** Production Streamlit UI")

        st.divider()
        st.markdown("### 🎯 Supported Sentiments")
        st.markdown(
            "- 🟢 **Positive**\n"
            "- 🔴 **Negative**\n"
            "- ⚪ **Neutral**"
        )
        st.divider()
        st.caption("Academic Minor Project | Deadline: 11 Sept 2026")


def main():
    """Main Streamlit application layout."""
    init_page()
    render_sidebar()

    st.title("🧠 AI Sentiment Intelligence")
    st.markdown("### Stage 2 Dashboard: Data Engineering & NLP Preprocessing Engine")

    # Navigation Tabs
    tab_nlp, tab_data, tab_charts, tab_arch = st.tabs([
        "🔬 Interactive NLP Engine",
        "📊 Dataset & Quality Audit",
        "📈 Exploratory Visualizations",
        "🏛️ System Architecture",
    ])

    # ==========================================================================
    # TAB 1: Interactive NLP Preprocessor
    # ==========================================================================
    with tab_nlp:
        st.markdown("#### Test the Real-time NLP Preprocessing Pipeline")
        st.markdown(
            "Type or paste any text below to observe the step-by-step transformation "
            "performed by the NLP engine (contraction expansion, normalization, "
            "negation preservation, and lemmatization)."
        )

        example_sentences = [
            "I don't like this phone, it's NOT good at all and the battery died!",
            "Customer support was quick, friendly, and resolved my issue in minutes.",
            "The device was delivered yesterday afternoon as scheduled.",
            "Check OUT: https://test.com! Email: help@site.com. <b>AWESOME</b> app... loved it!",
        ]

        selected_example = st.selectbox(
            "Or pick an example sentence to test:",
            ["(Custom input)"] + example_sentences,
        )

        default_text = (
            selected_example
            if selected_example != "(Custom input)"
            else "I didn't enjoy this movie, but the cinematography wasn't bad!"
        )

        user_input = st.text_area("Input Text:", value=default_text, height=100)

        col_opts1, col_opts2, col_opts3 = st.columns(3)
        with col_opts1:
            opt_contractions = st.checkbox("Expand Contractions", value=True)
        with col_opts2:
            opt_stopwords = st.checkbox("Filter Stopwords (Preserve Negations)", value=True)
        with col_opts3:
            opt_lemmatize = st.checkbox("Apply WordNet Lemmatization", value=True)

        if st.button("⚡ Run Preprocessing Pipeline", type="primary"):
            if user_input.strip():
                # Step-by-step execution
                step_contraction = expand_contractions(user_input) if opt_contractions else user_input
                step_basic = clean_text_basic(step_contraction, remove_punct=True)
                tokens_raw = tokenize_text(step_basic)
                tokens_filtered = remove_stopwords(tokens_raw) if opt_stopwords else tokens_raw
                tokens_lemmatized = lemmatize_tokens(tokens_filtered) if opt_lemmatize else tokens_filtered
                final_clean = " ".join(tokens_lemmatized)

                st.divider()
                st.markdown("##### 🏁 Preprocessing Result")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.info(f"**Original Text:**\n\n{user_input}")
                with col_res2:
                    st.success(f"**Cleaned Feature Text (TF-IDF Ready):**\n\n`{final_clean}`")

                with st.expander("🔍 Step-by-Step Pipeline Inspection", expanded=True):
                    st.markdown(f"1. **Contraction Expansion:** `{step_contraction}`")
                    st.markdown(f"2. **Lowercasing & Special Characters:** `{step_basic}`")
                    st.markdown(f"3. **Tokenization:** `{tokens_raw}`")
                    st.markdown(f"4. **Controlled Stopwords (Negations Retained):** `{tokens_filtered}`")
                    st.markdown(f"5. **WordNet Lemmatization:** `{tokens_lemmatized}`")
            else:
                st.warning("Please enter some text to preprocess.")

    # ==========================================================================
    # TAB 2: Dataset & Quality Audit
    # ==========================================================================
    with tab_data:
        st.markdown("#### Dataset Overview & Quality Audit")

        raw_path = get_default_dataset_path()
        st.caption(f"Active raw dataset: `{raw_path.resolve()}`")

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

            # Data tables
            st.divider()
            col_tbl1, col_tbl2 = st.columns(2)
            with col_tbl1:
                st.markdown("##### 📄 Raw Dataset Preview")
                st.dataframe(raw_df.head(10), use_container_width=True)

            with col_tbl2:
                st.markdown("##### 🧼 Cleaned Dataset Preview")
                if config.CLEANED_DATA_FILE.exists():
                    clean_df = pd.read_csv(config.CLEANED_DATA_FILE)
                    st.dataframe(clean_df.head(10), use_container_width=True)
                else:
                    st.info("Run `python run_stage2.py` to generate the cleaned dataset.")

            # Quality report
            if config.DATA_QUALITY_REPORT_TXT.exists():
                with st.expander("📋 Data Quality Audit Report", expanded=False):
                    with open(config.DATA_QUALITY_REPORT_TXT, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="text")

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # ==========================================================================
    # TAB 3: Exploratory Visualizations
    # ==========================================================================
    with tab_charts:
        st.markdown("#### Exploratory Data Visualizations (Stage 2)")
        st.caption("Visualizations generated directly from actual dataset execution.")

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            st.markdown("##### 1. Sentiment Class Distribution")
            if config.SENTIMENT_DIST_PLOT.exists():
                st.image(str(config.SENTIMENT_DIST_PLOT), use_container_width=True)
            else:
                st.info("Chart not found. Run `python run_stage2.py` to generate it.")

        with col_fig2:
            st.markdown("##### 2. Text Length & Word Count Distribution")
            if config.TEXT_LENGTH_PLOT.exists():
                st.image(str(config.TEXT_LENGTH_PLOT), use_container_width=True)
            else:
                st.info("Chart not found. Run `python run_stage2.py` to generate it.")

        st.divider()
        st.markdown("##### 3. Word Frequency Analysis by Sentiment Class")
        if config.WORD_FREQ_PLOT.exists():
            st.image(str(config.WORD_FREQ_PLOT), use_container_width=True)
        else:
            st.info("Chart not found. Run `python run_stage2.py` to generate it.")

    # ==========================================================================
    # TAB 4: System Architecture
    # ==========================================================================
    with tab_arch:
        st.markdown("#### End-to-End System Architecture")
        st.code("""
┌─────────────────────────┐
│     Raw Dataset CSV     │  (data/raw/demo_dataset.csv or dataset.csv)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Data Validation      │  (src/data_loader.py: missing, duplicate, schema checks)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     Data Cleaning       │  (src/data_cleaning.py: safe_str, lowercase, urls, html, emails)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   NLP Preprocessing     │  (src/preprocessing.py: contractions, tokens, controlled stops, lemmas)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Cleaned Dataset CSV   │  (data/processed/cleaned_dataset.csv: text, sentiment, clean_text)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   TF-IDF Vectorizer     │  [Stage 3: Feature Extraction]
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Classifier Models     │  [Stage 3: Logistic Regression, Naive Bayes, Linear SVM]
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Model Evaluation & UI   │  [Stages 4-6: Confusion Matrix, Metrics, Real-time Web UI]
└─────────────────────────┘
        """, language="text")


if __name__ == "__main__":
    main()
