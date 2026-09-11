"""Reports Component for SentimentAI (SentimentLab).

Organizes technical outputs:
- Classification reports
- Evaluation results
- Confusion matrices
- Error analysis
- Model comparison
- Downloadable reports
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import config


def render_reports_page():
    """Render the technical Reports page."""
    st.markdown("## ▤ Reports")
    st.caption("Access scikit-learn classification reports, confusion heatmaps, error analyses, and test benchmarking records.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Evaluation & Benchmarks",
        "Classification Reports",
        "Error Analysis",
        "Model Comparison",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Evaluation Results
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("#### Test Set Evaluation Metrics")
        if config.EVALUATION_RESULTS_CSV.exists():
            eval_df = pd.read_csv(config.EVALUATION_RESULTS_CSV)
            st.dataframe(eval_df, use_container_width=True)

            csv_data = config.EVALUATION_RESULTS_CSV.read_bytes()
            st.download_button(
                label="Download Evaluation Results (CSV)",
                data=csv_data,
                file_name="model_evaluation_results.csv",
                mime="text/csv",
                key="btn_dl_eval_csv",
            )
        else:
            st.info("Evaluation results file not found.")

    # -------------------------------------------------------------------------
    # TAB 2: Classification Reports
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("#### Scikit-Learn Classification Reports")
        reports_dir = config.CLASSIFICATION_REPORTS_DIR
        if reports_dir.exists():
            for r_file in sorted(reports_dir.glob("*_report.txt")):
                label = r_file.stem.replace("_report", "").replace("_", " ").title()
                with st.expander(f"Report: {label}", expanded=True):
                    with open(r_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.code(content, language="text")
        else:
            st.info("Classification reports not found.")

    # -------------------------------------------------------------------------
    # TAB 3: Error Analysis
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("#### Misclassified Test Samples")
        if config.ERROR_ANALYSIS_CSV.exists():
            err_df = pd.read_csv(config.ERROR_ANALYSIS_CSV)
            models = ["All"] + sorted(err_df["model"].unique().tolist())
            chosen_m = st.selectbox("Filter Errors by Model:", models, key="rep_err_sel")
            filtered_err = err_df if chosen_m == "All" else err_df[err_df["model"] == chosen_m]

            st.caption(f"Showing **{len(filtered_err)}** misclassified records.")
            st.dataframe(filtered_err, use_container_width=True)

            st.download_button(
                label="Download Error Analysis (CSV)",
                data=config.ERROR_ANALYSIS_CSV.read_bytes(),
                file_name="sentiment_error_analysis.csv",
                mime="text/csv",
                key="btn_dl_err_csv",
            )
        else:
            st.info("Error analysis file not found.")

    # -------------------------------------------------------------------------
    # TAB 4: Model Comparison
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("#### Side-by-Side Model Comparison")
        if config.MODEL_COMPARISON_CSV.exists():
            comp_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)
            st.dataframe(comp_df, use_container_width=True)
        elif config.EVALUATION_RESULTS_CSV.exists():
            st.dataframe(pd.read_csv(config.EVALUATION_RESULTS_CSV), use_container_width=True)
        else:
            st.info("Model comparison data not found.")