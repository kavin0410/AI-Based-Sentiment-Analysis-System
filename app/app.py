"""SentimentLab Web Application.

An ultra-premium AI SaaS platform for sentiment intelligence:
- Brand: 
    SentimentLab
    Understand the emotion behind every word.
    ● System Online
- Clean, continuous navigation without category labels:
    Overview
    Analyze
    Insights
    History
    NLP Explorer
    Model Intelligence
    Data
    Reports
    Settings
- Global command search palette ("Search SentimentLab...")
- Cyber Neural Video Background
- Zero development stage references anywhere in the UI.
"""

import base64
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Setup sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
from components.overview import render_overview_page
from components.analyze import render_analyze_page
from components.intelligence import render_insights_page
from components.history import render_history_page
from components.nlp_engine import render_nlp_engine_page
from components.model_lab import render_model_lab_page
from components.data_quality import render_data_quality_page
from components.reports import render_reports_page
from components.system import render_system_page
from src.model_loader import validate_model_artifacts
from src.predict import PredictionHistoryManager


@st.cache_data
def get_background_assets():
    """Encode video and poster image to base64 for smooth background playback."""
    poster_path = APP_DIR / "static" / "bg_poster.jpg"
    video_path = APP_DIR / "static" / "bg_video_optimized.mp4"

    poster_b64 = ""
    if poster_path.exists():
        poster_b64 = base64.b64encode(poster_path.read_bytes()).decode("utf-8")

    video_b64 = ""
    if video_path.exists():
        video_b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")

    return poster_b64, video_b64


def render_background_layers():
    """Inject background video and ambient neural vignette overlay."""
    poster_b64, video_b64 = get_background_assets()

    video_html = ""
    if video_b64:
        video_html = f"""
        <video autoplay loop muted playsinline poster="data:image/jpeg;base64,{poster_b64}">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        """
    elif poster_b64:
        video_html = f"""
        <div style="width: 100vw; height: 100vh; background: url('data:image/jpeg;base64,{poster_b64}') center/cover no-repeat; opacity: 0.35;"></div>
        """

    st.markdown(
        f"""
        <div id="bg-video-container">
            {video_html}
        </div>
        <div id="bg-video-overlay"></div>
        """,
        unsafe_allow_html=True,
    )


def init_page():
    """Initialize page metadata, layout, background layers, and custom styling."""
    st.set_page_config(
        page_title="SentimentLab | Understand the emotion behind every word",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css_path = APP_DIR / "static" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    render_background_layers()


def init_session_state():
    """Initialize persistent session managers and default routing."""
    if "history_manager" not in st.session_state:
        st.session_state["history_manager"] = PredictionHistoryManager(
            max_limit=config.PREDICTION_HISTORY_LIMIT
        )
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"


def render_sidebar():
    """Render the ultra-clean, continuous navigation sidebar without category headers."""
    nav_options = [
        "Overview",
        "Analyze",
        "Insights",
        "History",
        "NLP Explorer",
        "Model Intelligence",
        "Data",
        "Reports",
        "Settings",
    ]

    # Process any pending page navigation request
    if "target_page" in st.session_state and st.session_state["target_page"] in nav_options:
        st.session_state["active_page"] = st.session_state.pop("target_page")

    with st.sidebar:
        # Branding Header
        st.markdown(
            """
            <div class="brand-container">
                <div class="brand-title">
                    SentimentLab
                </div>
                <div class="brand-subtitle">
                    Understand the emotion behind every word.
                </div>
                <div class="status-pill">
                    <span class="status-dot"></span> System Online
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # Global Search / Command Bar
        search_cmd = st.text_input(
            "Search",
            placeholder="Search SentimentLab... (Ctrl+K)",
            key="global_search_input",
            label_visibility="collapsed",
        )

        # Handle search bar selection if matched
        if search_cmd.strip():
            matched = [opt for opt in nav_options if search_cmd.strip().lower() in opt.lower()]
            if matched and matched[0] != st.session_state.get("active_page"):
                st.session_state["active_page"] = matched[0]

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # Determine current index
        current_page = st.session_state.get("active_page", "Overview")
        current_index = nav_options.index(current_page) if current_page in nav_options else 0

        # Callback function for widget selection
        def on_nav_change():
            st.session_state["active_page"] = st.session_state["_sidebar_radio"]

        # Single continuous navigation list
        selected = st.radio(
            label="Navigation Menu",
            options=nav_options,
            index=current_index,
            key="_sidebar_radio",
            on_change=on_nav_change,
            label_visibility="collapsed",
        )

        st.session_state["active_page"] = selected

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="padding: 0 0.5rem; font-size: 0.75rem; color: #64748b; line-height: 1.6;">
                SentimentLab Platform v2.5.0<br/>
                All systems operational.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state["active_page"]


def render_top_header():
    """Render clean global page top banner."""
    st.markdown(
        """
        <div class="top-header">
            <div class="header-title-wrap">
                <h1>SentimentLab</h1>
                <p>Understand the emotion behind every word.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Application controller."""
    init_page()
    init_session_state()
    selected_page = render_sidebar()
    render_top_header()

    # Route based on navigation choice
    if selected_page == "Overview":
        render_overview_page()
    elif selected_page == "Analyze":
        render_analyze_page()
    elif selected_page == "Insights":
        render_insights_page()
    elif selected_page == "History":
        render_history_page()
    elif selected_page == "NLP Explorer":
        render_nlp_engine_page()
    elif selected_page == "Model Intelligence":
        render_model_lab_page()
    elif selected_page == "Data":
        render_data_quality_page()
    elif selected_page == "Reports":
        render_reports_page()
    elif selected_page == "Settings":
        render_system_page()


if __name__ == "__main__":
    main()