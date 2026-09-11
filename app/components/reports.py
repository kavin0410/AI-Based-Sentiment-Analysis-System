"""Reports Component for SentimentLab — Premium Light UI.

Report center:
- Evaluation & Benchmarks
- Classification Reports
- Error Analysis
- Model Comparison
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import config


def render_reports_page():
    """Render the Reports page."""
    st.markdown(
        """
        <div class="sl-page-header">
            <div class="sl-page-eyebrow">TECHNICAL REPORTS</div>
            <h1 class="sl-page-title">Reports</h1>
            <p class="sl-page-subtitle">
                Classification reports, confusion heatmaps, error analyses, and test benchmarks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Evaluation & Benchmarks",
        "Classification Reports",
        "Error Analysis",
        "Model Comparison",
    ])

    # ------------------------------------------------------------------
    # TAB 1: Evaluation Results
    # ------------------------------------------------------------------
    with tab1:
        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Test Set Evaluation Metrics</div>',
            unsafe_allow_html=True,
        )
        if config.EVALUATION_RESULTS_CSV.exists():
            eval_df = pd.read_csv(config.EVALUATION_RESULTS_CSV)
            st.dataframe(eval_df, use_container_width=True)

            st.download_button(
                label="Download Evaluation Results (CSV)",
                data=config.EVALUATION_RESULTS_CSV.read_bytes(),
                file_name="model_evaluation_results.csv",
                mime="text/csv",
                key="btn_dl_eval",
            )
        else:
            st.info("Evaluation results file not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TAB 2: Classification Reports
    # ------------------------------------------------------------------
    with tab2:
        st.markdown(
            '<div class="sl-section-label">Scikit-Learn Classification Reports</div>',
            unsafe_allow_html=True,
        )
        reports_dir = config.CLASSIFICATION_REPORTS_DIR
        if reports_dir.exists():
            for r_file in sorted(reports_dir.glob("*_report.txt")):
                label = r_file.stem.replace("_report", "").replace("_", " ").title()
                with st.expander(f"Report: {label}", expanded=True):
                    st.markdown('<div class="sl-card sl-card-sm">', unsafe_allow_html=True)
                    with open(r_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.code(content, language="text")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Classification reports not found.")

    # ------------------------------------------------------------------
    # TAB 3: Error Analysis
    # ------------------------------------------------------------------
    with tab3:
        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Misclassified Test Samples</div>',
            unsafe_allow_html=True,
        )
        if config.ERROR_ANALYSIS_CSV.exists():
            err_df = pd.read_csv(config.ERROR_ANALYSIS_CSV)
            models = ["All"] + sorted(err_df["model"].unique().tolist())
            chosen_m = st.selectbox("Filter by Model:", models, key="err_model_sel")
            filtered_err = err_df if chosen_m == "All" else err_df[err_df["model"] == chosen_m]

            st.caption(f"Showing **{len(filtered_err)}** misclassified records.")
            st.dataframe(filtered_err, use_container_width=True)

            st.download_button(
                label="Download Error Analysis (CSV)",
                data=config.ERROR_ANALYSIS_CSV.read_bytes(),
                file_name="sentiment_error_analysis.csv",
                mime="text/csv",
                key="btn_dl_err",
            )
        else:
            st.info("Error analysis file not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TAB 4: Model Comparison
    # ------------------------------------------------------------------
    with tab4:
        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Side-by-Side Model Comparison</div>',
            unsafe_allow_html=True,
        )
        if config.MODEL_COMPARISON_CSV.exists():
            comp_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)
            st.dataframe(comp_df, use_container_width=True)
        elif config.EVALUATION_RESULTS_CSV.exists():
            st.dataframe(pd.read_csv(config.EVALUATION_RESULTS_CSV), use_container_width=True)
        else:
            st.info("Model comparison data not found.")
        st.markdown("</div>", unsafe_allow_html=True)