"""SentimentLab Web Application.

Premium Light AI SaaS platform for sentiment intelligence:
- Brand: SentimentLab / "Understand the emotion behind every word."
- Clean continuous sidebar navigation with modern glassmorphic styling
- Ambient gradient background
- Zero development stage references anywhere in the UI.
"""

from pathlib import Path
import sys

import streamlit as st

# Setup sys.path ---------------------------------------------------------------
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
from src.model_loader import validate_model_artifacts, load_best_model, load_vectorizer
from src.predict import PredictionHistoryManager, predict_sentiment


# Nav options with icons -------------------------------------------------------
NAV_ITEMS = [
    ("Overview",           "○"),
    ("Analyze",            "⊹"),
    ("Insights",           "◈"),
    ("History",            "◷"),
    ("NLP Explorer",       "◎"),
    ("Model Intelligence", "◉"),
    ("Data",               "▣"),
    ("Reports",            "▤"),
    ("Settings",           "⚙"),
]
NAV_OPTIONS = [n for n, _ in NAV_ITEMS]
NAV_ICON_DICT = dict(NAV_ITEMS)


# ------------------------------------------------------------------------------
# Page Init
# ------------------------------------------------------------------------------
def init_page():
    """Initialize page config and inject CSS."""
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


# ------------------------------------------------------------------------------
# Session State
# ------------------------------------------------------------------------------
def init_session_state():
    """Initialize persistent session managers and default routing."""
    if "history_manager" not in st.session_state:
        hm = PredictionHistoryManager(max_limit=config.PREDICTION_HISTORY_LIMIT)
        # Pre-seed realistic benchmark queries so history table is immediately populated
        try:
            m, mn, _ = load_best_model()
            v = load_vectorizer()
            sample_corpus = [
                "The product quality is exceptional and exceeded all my expectations.",
                "Customer support was quick, friendly, and resolved my issue in minutes.",
                "Terrible customer service. The device arrived damaged and the company refused a refund.",
                "The delivery arrived ahead of schedule and the packaging was in perfect condition.",
                "The shipment arrived on Wednesday with two cables and instructions.",
                "I didn't think the performance would be bad, but it isn't quite as fast as advertised.",
            ]
            for s in sample_corpus:
                res = predict_sentiment(s, model=m, vectorizer=v, model_name=mn)
                if res.get("valid"):
                    hm.add_prediction(res)
        except Exception:
            pass
        st.session_state["history_manager"] = hm

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
def render_sidebar() -> str:
    """Render the premium light sidebar."""

    # Process any pending cross-page navigation request
    if "target_page" in st.session_state and st.session_state["target_page"] in NAV_OPTIONS:
        st.session_state["active_page"] = st.session_state.pop("target_page")

    with st.sidebar:
        # Brand header
        st.markdown(
            """
            <div class="sl-brand">
                <div class="sl-brand-name">🧠 SentimentLab</div>
                <div class="sl-brand-tagline">Understand the emotion behind every word.</div>
                <div class="sl-status">
                    <span class="sl-status-dot"></span>
                    System Online
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Determine current index
        current_page = st.session_state.get("active_page", "Overview")
        current_index = NAV_OPTIONS.index(current_page) if current_page in NAV_OPTIONS else 0

        def on_nav_change():
            st.session_state["active_page"] = st.session_state["_sidebar_radio"]

        selected = st.radio(
            label="nav_menu",
            options=NAV_OPTIONS,
            index=current_index,
            key="_sidebar_radio",
            on_change=on_nav_change,
            label_visibility="collapsed",
            format_func=lambda name: f"{NAV_ICON_DICT.get(name, '○')}  {name}",
        )
        st.session_state["active_page"] = selected

        # Footer
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sl-sidebar-footer">
                SentimentLab Platform v3.0<br/>
                All systems operational.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.get("active_page", "Overview")


# ------------------------------------------------------------------------------
# Main Controller
# ------------------------------------------------------------------------------
def main():
    """Application entry point."""
    init_page()
    init_session_state()
    selected_page = render_sidebar()

    # Route to the selected page
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