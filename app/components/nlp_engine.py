"""NLP Explorer Component for SentimentAI (SentimentLab).

Interactive demonstration of the custom NLP pipeline:
- original text
- cleaned text
- contraction expansion
- tokenization
- stopword handling
- negation preservation
- lemmatization
- final processed text
"""

import streamlit as st
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
    """Render the interactive NLP Explorer page."""
    st.markdown("## ◎ NLP Explorer")
    st.caption("Interactively trace raw, noisy text through our multi-step natural language transformation pipeline.")

    # -------------------------------------------------------------------------
    # 1. Interactive Input Area
    # -------------------------------------------------------------------------
    demo_samples = [
        "I don't like this device, it isn't reliable and the battery died immediately!",
        "Customer service was prompt, polite, and addressed all my questions within minutes.",
        "The shipment arrived on Wednesday afternoon. The box contains three replacement cables.",
        "I didn't think the film was bad, but the pacing wasn't entirely smooth.",
    ]

    selected_sample = st.selectbox(
        "Choose an example or enter your own text below:",
        ["(Custom Text)"] + demo_samples,
        key="nlp_explorer_sample_sel",
    )

    default_val = selected_sample if selected_sample != "(Custom Text)" else "I haven't had a worse experience with customer support; it wasn't helpful!"
    user_input = st.text_area("Input Text to Trace:", value=default_val, height=85, key="nlp_explorer_input_area")

    if not user_input.strip():
        st.info("Please enter or select text above to visualize the pipeline.")
        return

    # -------------------------------------------------------------------------
    # 2. Pipeline Execution Steps
    # -------------------------------------------------------------------------
    raw_str = user_input.strip()

    # Step 1: Contractions
    s1_contractions = expand_contractions(raw_str)

    # Step 2: Cleaning & Normalization
    s2_cleaned = clean_text_basic(s1_contractions)

    # Step 3: Tokenization
    s3_tokens = tokenize_text(s2_cleaned)

    # Step 4: Stopword Removal (Negations Preserved)
    s4_no_stops = remove_stopwords(s3_tokens, preserve_negations=True)

    # Step 5: Lemmatization
    s5_lemmatized = lemmatize_tokens(s4_no_stops)

    # Step 6: Final Processed String
    s6_final = " ".join(s5_lemmatized)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Pipeline Transformation Stages")

    with st.expander("1. Original Text", expanded=True):
        st.code(raw_str, language="text")

    with st.expander("2. Contraction Expansion", expanded=True):
        st.caption("Expands contractions to preserve sentiment predicates (e.g., *don't* → *do not*, *wasn't* → *was not*).")
        st.code(s1_contractions, language="text")

    with st.expander("3. Cleaning & Normalization", expanded=True):
        st.caption("Converts characters to lowercase, strips URLs, HTML markup, and non-alphanumeric noise.")
        st.code(s2_cleaned, language="text")

    with st.expander("4. Tokenization", expanded=True):
        st.caption("Splits the normalized character sequence into discrete linguistic tokens using NLTK.")
        st.write(s3_tokens)

    with st.expander("5. Stopword Handling & Negation Preservation", expanded=True):
        st.caption("Filters domain-neutral stopwords while strictly retaining critical negation words (*not, no, never*).")
        st.write(s4_no_stops)

    with st.expander("6. WordNet Lemmatization", expanded=True):
        st.caption("Reduces morphological variants to their canonical dictionary root words (e.g., *running* → *run*).")
        st.write(s5_lemmatized)

    with st.expander("7. Final Processed Text", expanded=True):
        st.caption("Reconstituted clean token string prepared for TF-IDF feature matrix conversion (570 features).")
        st.code(s6_final, language="text")