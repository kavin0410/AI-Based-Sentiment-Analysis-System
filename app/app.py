"""SentimentLab Web Application.

Premium Light AI SaaS platform for sentiment intelligence:
- Brand: SentimentLab / "Understand the emotion behind every word."
- Clean continuous sidebar navigation with custom button-based nav
- Ambient gradient background (no video — light theme)
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
from src.model_loader import validate_model_artifacts
from src.predict import PredictionHistoryManager


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

    # Inject ambient pink orb div + hide default streamlit chrome
    st.markdown(
        """
        <div class="sl-orb-pink"></div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# Session State
# ------------------------------------------------------------------------------
def init_session_state():
    """Initialize persistent session managers and default routing."""
    if "history_manager" not in st.session_state:
        st.session_state["history_manager"] = PredictionHistoryManager(
            max_limit=config.PREDICTION_HISTORY_LIMIT
        )
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
def render_sidebar() -> str:
    """Render the premium light sidebar with button-based navigation."""

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

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        # Search bar
        search_val = st.text_input(
            "Search",
            placeholder="Search SentimentLab... (Ctrl+K)",
            key="global_search",
            label_visibility="collapsed",
        )
        if search_val.strip():
            matched = [o for o in NAV_OPTIONS if search_val.strip().lower() in o.lower()]
            if matched and matched[0] != st.session_state.get("active_page"):
                st.session_state["active_page"] = matched[0]
                st.rerun()

        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

        # Navigation buttons
        current_page = st.session_state.get("active_page", "Overview")
        for page_name, icon in NAV_ITEMS:
            is_active = current_page == page_name
            label = f"{icon}  {page_name}"
            # Use a unique key per nav button
            btn_key = f"nav_btn_{page_name.replace(' ', '_')}"
            if is_active:
                # Active item rendered as colored HTML block + disabled button approach
                st.markdown(
                    f"""
                    <div class="sl-nav-item active">
                        <span class="sl-nav-icon">{icon}</span>
                        {page_name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                if st.button(label, key=btn_key, use_container_width=True):
                    st.session_state["active_page"] = page_name
                    st.rerun()

        # Footer
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sl-sidebar-footer">
                SentimentLab v3.0<br/>
                Premium AI SaaS Edition
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