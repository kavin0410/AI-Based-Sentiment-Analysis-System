"""Reports Component for AI Sentiment Intelligence (Stage 6).

Provides scikit-learn text classification reports display and downloads for each evaluated model.
"""

from pathlib import Path
import sys

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config


def render_reports_page():
    """Render the Classification Reports page."""
    st.markdown("## 📋 Classification Reports")
    st.caption("Detailed scikit-learn precision, recall, F1-score, and support reports evaluated on the test split.")

    reports_dir = config.CLASSIFICATION_REPORTS_DIR
    if not reports_dir.exists():
        st.warning("Classification reports not found. Please execute Stage 4 (`python run_stage4.py`).")
        return

    report_files = sorted(reports_dir.glob("*_report.txt"))
    if not report_files:
        st.info("No individual report files found.")
        return

    for report_file in report_files:
        model_name = report_file.stem.replace("_report", "").replace("_", " ").title()
        with st.expander(f"📄 Classification Report: {model_name}", expanded=True):
            with open(report_file, "r", encoding="utf-8") as f:
                content = f.read()
            st.code(content, language="text")
