"""Streamlit Web Application for AI Sentiment Intelligence Platform (Stage 6).

Transforming technical demonstration capabilities into an AI product experience:
- 🏠 Overview: Executive landing page, KPI metrics, dynamic Best Model card, transparent dataset health disclaimer.
- ⚡ Analyze: Real-time predictor with confidence scores, NLP pipeline transformation inspector, session history & CSV export.
- 📊 Intelligence: Unified performance analytics, sentiment distributions, confusion matrices, and error intelligence.
- 🧠 Model Lab: Model specifications, hyperparameter inspection, and automated selection logic explanation.
- 🔬 NLP Engine: Visual storytelling & interactive 12-step NLP preprocessing pipeline simulator (Negation Preservation).
- 🛡️ Data & Quality: Dataset audit, text statistics, raw vs cleaned corpus previews, downloadable quality reports.
- 📋 Reports: Scikit-learn classification text report viewer.
- 🏛️ Architecture: End-to-end data flow & pipeline diagram.
"""

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from app.components.architecture import render_architecture_page
from app.components.analyze import render_analyze_page
from app.components.data_quality import render_data_quality_page
from app.components.intelligence import render_intelligence_page
from app.components.model_lab import render_model_lab_page
from app.components.nlp_engine import render_nlp_engine_page
from app.components.overview import render_overview_page
from app.components.reports import render_reports_page
from src.model_loader import validate_model_artifacts
from src.predict import PredictionHistoryManager


def init_page():
    """Initialize Streamlit page configuration and dark theme visual branding."""
    st.set_page_config(
        page_title="AI Sentiment Intelligence Platform",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_session_state():
    """Initialize Streamlit session state for prediction history and text inputs."""
    if "history_manager" not in st.session_state:
        st.session_state["history_manager"] = PredictionHistoryManager(
            max_limit=config.PREDICTION_HISTORY_LIMIT
        )
    if "user_text_input" not in st.session_state:
        st.session_state["user_text_input"] = ""


def render_header():
    """Render global application header banner with system status indicator."""
    val_report = validate_model_artifacts()
    status_indicator = "🟢 ● AI Engine Online" if val_report["valid"] else "🔴 ● AI Engine Degraded"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 10px;">
            <div>
                <h1 style="margin: 0; padding: 0;">🧠 AI Sentiment Intelligence</h1>
                <p style="margin: 0; color: #888; font-size: 1.1rem;">AI-Based Sentiment Analysis Platform | <i>Understand sentiment through machine learning</i></p>
            </div>
            <div style="text-align: right; background: #1e222d; padding: 8px 16px; border-radius: 8px; border: 1px solid #333;">
                <span style="font-weight: bold; color: {'#2ecc71' if val_report['valid'] else '#e74c3c'};">{status_indicator}</span><br/>
                <small style="color: #aaa;">Best Model: {val_report.get('best_model_name', 'Loaded')}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_sidebar():
    """Render sidebar branding, operational system status, and project stage tracker."""
    with st.sidebar:
        st.markdown("## 🧠 AI Sentiment Intelligence")
        st.caption("AI-Based Sentiment Analysis System Using Machine Learning")
        st.divider()

        st.markdown("### 🟢 System Status")
        st.markdown(
            "- 🟢 **Online:** Core Engine Active\n"
            "- 🟢 **Models:** 3 Models Loaded\n"
            "- 🟢 **Predictor:** Real-Time Ready"
        )

        st.divider()
        st.markdown("### 📌 Project Progress")
        st.success("✅ **Stage 1:** Foundation")
        st.success("✅ **Stage 2:** Dataset & NLP Engine")
        st.success("✅ **Stage 3:** Feature Engineering & ML")
        st.success("✅ **Stage 4:** Evaluation & Performance")
        st.success("✅ **Stage 5:** Real-Time Prediction")
        st.success("✅ **Stage 6:** Sentiment Intelligence Experience")
        st.success("🟢 **Stage 7:** Finalized & Submission Ready")

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
    """Main Streamlit application entry point."""
    init_page()
    init_session_state()
    render_header()
    render_sidebar()

    # Primary Navigation Structure
    (
        tab_overview,
        tab_analyze,
        tab_intel,
        tab_model_lab,
        tab_nlp,
        tab_quality,
        tab_reports,
        tab_arch,
    ) = st.tabs([
        "🏠 Overview",
        "⚡ Analyze",
        "📊 Intelligence",
        "🧠 Model Lab",
        "🔬 NLP Engine",
        "🛡️ Data & Quality",
        "📋 Reports",
        "🏛️ Architecture",
    ])

    with tab_overview:
        render_overview_page()

    with tab_analyze:
        render_analyze_page()

    with tab_intel:
        render_intelligence_page()

    with tab_model_lab:
        render_model_lab_page()

    with tab_nlp:
        render_nlp_engine_page()

    with tab_quality:
        render_data_quality_page()

    with tab_reports:
        render_reports_page()

    with tab_arch:
        render_architecture_page()


if __name__ == "__main__":
    main()
