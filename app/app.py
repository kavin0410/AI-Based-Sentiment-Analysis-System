"""Streamlit Web Application for AI Sentiment Intelligence.

A premium dark AI SaaS interface for sentiment analysis:
- Modern navigation grouped into: MAIN, AI, DATA, SYSTEM
- Seamless real-time classification, deep explainability, and analytics
- No development stages shown in the UI — capabilities-focused
"""

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Setup python path
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


def init_page():
    """Initialize page metadata, layout, and inject custom CSS."""
    st.set_page_config(
        page_title="AI Sentiment Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject premium custom CSS
    css_path = APP_DIR / "static" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """Ensure session state structures are present."""
    if "history_manager" not in st.session_state:
        st.session_state["history_manager"] = PredictionHistoryManager(
            max_limit=config.PREDICTION_HISTORY_LIMIT
        )
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "🏠 Overview"


def render_header():
    """Render global top app header bar."""
    val_report = validate_model_artifacts()
    is_online = val_report.get("valid", True)
    status_text = "AI ENGINE ONLINE" if is_online else "ENGINE DEGRADED"
    status_color = "#3fb950" if is_online else "#f85149"
    best_m = val_report.get("best_model_name", "Logistic Regression")

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; padding-bottom: 0.5rem; margin-bottom: 0.5rem;">
            <div>
                <h1 class="asi-header-title">🧠 AI Sentiment Intelligence</h1>
                <p class="asi-header-tagline">Understand what people feel through machine learning.</p>
            </div>
            <div style="text-align: right; background: rgba(22, 27, 34, 0.8); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="font-size: 0.8rem; font-weight: 700; color: {status_color}; letter-spacing: 0.05em;">
                    <span class="pulse-dot" style="background-color: {status_color}; box-shadow: 0 0 8px {status_color};"></span>● {status_text}
                </div>
                <div style="font-size: 0.75rem; color: #8b949e; margin-top: 3px;">
                    Active Model: <span style="color: #f0f6fc;">{best_m}</span>
                </div>
            </div>
        </div>
        <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 0.5rem 0 1.25rem 0;" />
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render compact, grouped navigation sidebar."""
    with st.sidebar:
        st.markdown("### 🧠 AI Sentiment")
        st.caption("Machine Learning Platform")

        st.markdown(
            """
            <div style="display: inline-flex; align-items: center; background: rgba(46, 160, 67, 0.12); padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.25);">
                <span class="pulse-dot"></span> ● LIVE SYSTEM
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Compact Grouped Navigation
        nav_options = [
            "🏠 Overview",
            "✨ Analyzer",
            "📊 Insights",
            "🕘 History",
            "🧠 NLP Explorer",
            "🤖 Model Lab",
            "🗄️ Data Center",
            "📑 Reports",
            "⚙️ System",
        ]

        st.caption("MAIN")
        choice_main = [opt for opt in nav_options[:4]]
        st.caption("AI & MODELS")
        choice_ai = [opt for opt in nav_options[4:6]]
        st.caption("DATA & REPORTS")
        choice_data = [opt for opt in nav_options[6:8]]
        st.caption("SYSTEM")
        choice_sys = [nav_options[8]]

        selected = st.radio(
            label="Navigation",
            options=nav_options,
            key="asi_nav_selection",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Supported Polarity Legend
        st.caption("SENTIMENT POLARITIES")
        st.markdown(
            """
            <div style="font-size: 0.8rem; line-height: 1.8;">
                <span style="color: #3fb950;">●</span> <strong>Positive</strong><br/>
                <span style="color: #c9d1d9;">●</span> <strong>Neutral</strong><br/>
                <span style="color: #f85149;">●</span> <strong>Negative</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("AI Sentiment Intelligence v2.0")

    return selected


def main():
    """Main application lifecycle controller."""
    init_page()
    init_session_state()
    render_header()
    selected_page = render_sidebar()

    # Route to selected capability
    if selected_page == "🏠 Overview":
        render_overview_page()
    elif selected_page == "✨ Analyzer":
        render_analyze_page()
    elif selected_page == "📊 Insights":
        render_insights_page()
    elif selected_page == "🕘 History":
        render_history_page()
    elif selected_page == "🧠 NLP Explorer":
        render_nlp_engine_page()
    elif selected_page == "🤖 Model Lab":
        render_model_lab_page()
    elif selected_page == "🗄️ Data Center":
        render_data_quality_page()
    elif selected_page == "📑 Reports":
        render_reports_page()
    elif selected_page == "⚙️ System":
        render_system_page()


if __name__ == "__main__":
    main()