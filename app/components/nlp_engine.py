"""NLP Engine Component for AI Sentiment Intelligence (Stage 6).

Provides an interactive demonstration and visual storytelling of the 12-step NLP preprocessing pipeline:
- Contraction expansion
- Lowercasing & noise removal
- Tokenization
- Controlled stopword filtering with Negation Preservation
- WordNet Lemmatization
"""

from pathlib import Path
import sys

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.data_cleaning import clean_text_basic
from src.preprocessing import (
    expand_contractions,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    tokenize_text,
)


def render_nlp_engine_page():
    """Render the Interactive NLP Engine page."""
    st.markdown("## 🔬 Interactive NLP Preprocessing Engine")
    st.caption("Step-by-step text normalization, contraction expansion, negation preservation, and WordNet lemmatization.")

    # 1. Negation Preservation Callout
    st.divider()
    st.markdown("### 🛑 Controlled Stopword Strategy: Negation Preservation")
    st.info(
        "💡 **Why Negation Preservation Matters:** Standard stopword lists strip words like *'not'*, *'no'*, and *'never'*. "
        "For example, converting *'not good'* to *'good'* completely flips sentiment! "
        "Our custom NLP engine explicitly retains **sentiment-critical negations**: "
        "`{'not', 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'without', 'against'}`."
    )

    # 2. Pipeline Storytelling Flow
    st.divider()
    st.markdown("### 🔄 12-Step NLP Transformation Pipeline")
    st.code(
        """
RAW TEXT
  │
  ├── 1. Safe String Conversion
  ├── 2. Contraction Expansion (don't -> do not, isn't -> is not)
  ├── 3. Lowercase Normalization
  ├── 4. Remove URLs & Web Links
  ├── 5. Remove HTML Tags & Entities (<b> -> empty)
  ├── 6. Remove Email Addresses
  ├── 7. Remove Special Characters (retaining letters & spaces)
  ├── 8. Extra Whitespace Collapse
  ├── 9. NLTK Word Tokenization
  ├── 10. Controlled Stopword Filtering (Strictly Retaining Negations)
  ├── 11. WordNet Lemmatization (running -> run, better -> good)
  └── 12. Rejoin Clean Space-Separated Token String
  │
TF-IDF VECTORIZER MATRIX (570 Features)
        """,
        language="text",
    )

    # 3. Interactive Preprocessing Pipeline Simulator
    st.divider()
    st.markdown("### 🧪 Test the Real-time Pipeline Simulator")

    example_sentences = [
        "I don't like this phone, it's NOT good at all and the battery died!",
        "Customer support was quick, friendly, and resolved my issue in minutes.",
        "The device was delivered yesterday afternoon as scheduled.",
        "Check OUT: https://test.com! Email: help@site.com. <b>AWESOME</b> app... loved it!",
    ]

    selected_example = st.selectbox(
        "Pick an example sentence to test:",
        ["(Custom input)"] + example_sentences,
        key="nlp_sim_select",
    )

    default_text = (
        selected_example
        if selected_example != "(Custom input)"
        else "I didn't enjoy this movie, but the cinematography wasn't bad!"
    )

    user_input = st.text_area("Input Text to Preprocess:", value=default_text, height=100, key="nlp_sim_input")

    col_opts1, col_opts2, col_opts3 = st.columns(3)
    with col_opts1:
        opt_contractions = st.checkbox("Expand Contractions", value=True, key="cb_contractions")
    with col_opts2:
        opt_stopwords = st.checkbox("Filter Stopwords (Preserve Negations)", value=True, key="cb_stopwords")
    with col_opts3:
        opt_lemmatize = st.checkbox("Apply WordNet Lemmatization", value=True, key="cb_lemmatize")

    if st.button("⚡ Run Preprocessing Pipeline", type="primary", key="btn_run_nlp_sim"):
        if user_input.strip():
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
                st.info(f"**Original Input Text:**\n\n{user_input}")
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
