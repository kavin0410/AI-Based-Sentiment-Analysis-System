"""NLP Explorer Component for SentimentLab — Premium Light UI.

Interactive 7-step NLP pipeline visualizer with floating glass nodes.
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


def _pipeline_node(step_num: int, title: str, description: str, output: str):
    """Render a single pipeline step as a floating glass node."""
    st.markdown(
        f"""
        <div class="sl-pipeline-node">
            <div style="display:flex;align-items:center;margin-bottom:4px;">
                <span class="sl-pipeline-step-num">{step_num}</span>
                <span class="sl-pipeline-step-label">{title}</span>
            </div>
            <div class="sl-pipeline-step-desc">{description}</div>
            <div class="sl-pipeline-output">{output}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _connector():
    """Render a connecting arrow between pipeline nodes."""
    st.markdown(
        '<div class="sl-pipeline-connector">\u2193</div>',
        unsafe_allow_html=True,
    )


def render_nlp_engine_page():
    """Render the interactive NLP Explorer page."""
    st.markdown(
        """
        <div class="sl-page-header">
            <div class="sl-page-eyebrow">LANGUAGE PROCESSING</div>
            <h1 class="sl-page-title">NLP Explorer</h1>
            <p class="sl-page-subtitle">
                Trace raw text through our multi-step natural language transformation pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Input Area
    # ------------------------------------------------------------------
    st.markdown('<div class="sl-card">', unsafe_allow_html=True)

    demo_samples = [
        "I don't like this device, it isn't reliable and the battery died immediately!",
        "Customer service was prompt, polite, and addressed all my questions within minutes.",
        "The shipment arrived on Wednesday afternoon. The box contains three replacement cables.",
        "I didn't think the film was bad, but the pacing wasn't entirely smooth.",
    ]

    selected_sample = st.selectbox(
        "Choose an example or enter your own text below:",
        ["(Custom Text)"] + demo_samples,
        key="nlp_sample_sel",
    )

    default_val = (
        selected_sample
        if selected_sample != "(Custom Text)"
        else "I haven't had a worse experience with customer support; it wasn't helpful!"
    )
    user_input = st.text_area(
        "Input Text",
        value=default_val,
        height=85,
        key="nlp_input",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not user_input.strip():
        st.info("Please enter or select text above to visualize the pipeline.")
        return

    # ------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------
    raw_str = user_input.strip()
    s1_contractions = expand_contractions(raw_str)
    s2_cleaned = clean_text_basic(s1_contractions)
    s3_tokens = tokenize_text(s2_cleaned)
    s4_no_stops = remove_stopwords(s3_tokens)
    s5_lemmatized = lemmatize_tokens(s4_no_stops)
    s6_final = " ".join(s5_lemmatized)

    st.markdown("<div class='sl-spacer-md'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sl-section-label">Pipeline Transformation</div>',
        unsafe_allow_html=True,
    )

    # Render pipeline nodes
    _pipeline_node(
        1, "Original Text",
        "Raw input text exactly as received.",
        raw_str,
    )
    _connector()

    _pipeline_node(
        2, "Contraction Expansion",
        "Expands contractions to preserve sentiment predicates (e.g., <em>don\u2019t</em> \u2192 <em>do not</em>).",
        s1_contractions,
    )
    _connector()

    _pipeline_node(
        3, "Cleaning & Normalization",
        "Lowercase conversion, URL/HTML stripping, non-alphanumeric noise removal.",
        s2_cleaned,
    )
    _connector()

    _pipeline_node(
        4, "Tokenization",
        "Splits normalized text into discrete linguistic tokens using NLTK.",
        " | ".join(s3_tokens) if s3_tokens else "(empty)",
    )
    _connector()

    _pipeline_node(
        5, "Stopword Handling & Negation Preservation",
        "Filters domain-neutral stopwords while retaining critical negation words (<em>not, no, never</em>).",
        " | ".join(s4_no_stops) if s4_no_stops else "(empty)",
    )
    _connector()

    _pipeline_node(
        6, "WordNet Lemmatization",
        "Reduces morphological variants to canonical root words (e.g., <em>running</em> \u2192 <em>run</em>).",
        " | ".join(s5_lemmatized) if s5_lemmatized else "(empty)",
    )
    _connector()

    _pipeline_node(
        7, "Final Processed Text",
        "Reconstituted clean token string prepared for TF-IDF feature matrix conversion (570 features).",
        s6_final if s6_final.strip() else "(empty)",
    )