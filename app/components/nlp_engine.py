"""NLP Explorer Component for AI Sentiment Intelligence.

Provides interactive step-by-step visual NLP pipeline walkthrough:
Raw Text -> Contraction Expansion -> Normalization -> Tokenization
-> Controlled Stopword Removal -> Negation Preservation -> Lemmatization -> Final Text
"""

import re
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
    st.markdown("## 🧠 NLP Explorer")
    st.caption("Inspect how unstructured human language is transformed into a clean feature space, step-by-step.")

    # -------------------------------------------------------------------------
    # 1. Negation Preservation Callout Card
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="asi-card" style="border-left: 3px solid #388bfd;">
            <div style="font-weight: 600; color: #58a6ff; font-size: 0.95rem;">💡 Controlled Stopword Removal with Negation Preservation</div>
            <div style="font-size: 0.85rem; color: #8b949e; margin-top: 4px;">
                Standard NLP pipelines blindly remove negation words such as <em>"not"</em>, <em>"no"</em>, and <em>"never"</em>,
                turning <strong>"not satisfied"</strong> into <strong>"satisfied"</strong> — totally reversing sentiment!
                Our custom pipeline preserves 12 essential negation tokens to protect sentiment polarity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Interactive Input
    # -------------------------------------------------------------------------
    st.markdown("### 🔬 Test the Preprocessing Pipeline")
    preset_examples = [
        "I don't like this phone, it's NOT good at all and the battery died!",
        "Customer support was quick, friendly, and resolved my issue in minutes.",
        "The device was delivered yesterday afternoon as scheduled.",
        "I didn't think the movie was bad, but the cinematography wasn't great either.",
    ]

    selected_ex = st.selectbox(
        "Choose an example or enter your own text below:",
        ["(Custom Text)"] + preset_examples,
        key="nlp_preset_sel",
    )

    default_val = selected_ex if selected_ex != "(Custom Text)" else "I haven't seen a better product all year; it isn't disappointing!"
    user_input = st.text_area("Input Text to Trace:", value=default_val, height=90, key="nlp_trace_input")

    if not user_input.strip():
        st.info("Enter text above to see the step-by-step transformation.")
        return

    # -------------------------------------------------------------------------
    # 3. Step-by-Step Transformations (Preserving real logic)
    # -------------------------------------------------------------------------
    raw_text = user_input.strip()

    # Step 1: Contraction Expansion
    step1_expanded = expand_contractions(raw_text)

    # Step 2: Normalization (clean_text_basic: lowercase, strip URLs, HTML, special chars)
    step2_normalized = clean_text_basic(step1_expanded)

    # Step 3: Tokenization
    step3_tokens = tokenize_text(step2_normalized)

    # Step 4: Controlled Stopword Removal (Negations preserved)
    step4_no_stops = remove_stopwords(step3_tokens, preserve_negations=True)

    # Step 5: Lemmatization
    step5_lemmatized = lemmatize_tokens(step4_no_stops)

    # Step 6: Final Clean String
    step6_final = " ".join(step5_lemmatized)

    st.markdown("#### Transformation Pipeline")

    # Step 0: Raw Text
    with st.expander("Step 0: Raw Input Text", expanded=True):
        st.code(raw_text, language="text")

    # Step 1: Contractions
    with st.expander("Step 1: Contraction Expansion", expanded=True):
        st.caption("Expands colloquial contractions (e.g., *don't* -> *do not*, *isn't* -> *is not*).")
        st.code(step1_expanded, language="text")

    # Step 2: Normalization
    with st.expander("Step 2: Normalization & Cleaning", expanded=True):
        st.caption("Converts to lowercase, removes URLs, HTML tags, special symbols, and excess whitespace.")
        st.code(step2_normalized, language="text")

    # Step 3: Tokenization
    with st.expander("Step 3: Tokenization", expanded=True):
        st.caption("Splits the normalized sentence into individual linguistic tokens using NLTK.")
        st.write(step3_tokens)

    # Step 4: Stopword Removal + Negation Preservation
    with st.expander("Step 4: Controlled Stopword Removal (Negations Kept)", expanded=True):
        st.caption("Filters non-informative stopwords while retaining sentiment-critical negations.")
        st.write(step4_no_stops)

    # Step 5: Lemmatization
    with st.expander("Step 5: WordNet Lemmatization", expanded=True):
        st.caption("Reduces inflected tokens to canonical base lemmas (e.g., *running* -> *run*).")
        st.write(step5_lemmatized)

    # Step 6: Final Text
    with st.expander("Step 6: Final Feature-Ready Representation", expanded=True):
        st.caption("The reconstructed clean string passed directly into the TF-IDF vectorizer (570 features).")
        st.code(step6_final, language="text")