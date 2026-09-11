"""Reports Component for AI Sentiment Intelligence.

Provides technical output review organized into tabs:
- Evaluation: Metrics table & best model overview
- Classification Reports: Text outputs from scikit-learn
- Confusion Matrices: Matrix heatmaps
- Error Analysis: Misclassified sample inspection with filtering
- Model Comparison: Side-by-side performance table
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import config


def render_reports_page():
    """Render the technical Reports page."""
    st.markdown("## 📑 Technical Reports")
    st.caption("Inspect and download the full suite of model validation, classification, error analysis, and benchmarking reports.")

    tab_eval, tab_class_rep, tab_cm, tab_err, tab_comp = st.tabs([
        "📊 Evaluation",
        "📄 Classification Reports",
        "🧩 Confusion Matrices",
        "🔍 Error Analysis",
        "⚖️ Model Comparison",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Evaluation Overview
    # -------------------------------------------------------------------------
    with tab_eval:
        st.markdown("#### Evaluation Summary")
        if config.BEST_MODEL_JSON.exists():
            with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
                best_info = json.load(f)
            st.json(best_info)
        else:
            st.info("best_model.json not found.")

        if config.EVALUATION_RESULTS_CSV.exists():
            st.markdown("##### Performance by Model")
            eval_df = pd.read_csv(config.EVALUATION_RESULTS_CSV)
            st.dataframe(eval_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Classification Reports
    # -------------------------------------------------------------------------
    with tab_class_rep:
        st.markdown("#### Scikit-Learn Classification Reports")
        reports_dir = config.CLASSIFICATION_REPORTS_DIR

        if reports_dir.exists():
            rep_files = sorted(reports_dir.glob("*_report.txt"))
            for r_file in rep_files:
                name = r_file.stem.replace("_report", "").replace("_", " ").title()
                with st.expander(f"Report: {name}", expanded=True):
                    with open(r_file, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="text")
        else:
            st.info("Classification reports directory not found.")

    # -------------------------------------------------------------------------
    # TAB 3: Confusion Matrices
    # -------------------------------------------------------------------------
    with tab_cm:
        st.markdown("#### Confusion Matrix Visualizations")
        cm_dir = config.RESULTS_PATH / "confusion_matrices"

        col1, col2, col3 = st.columns(3)
        cm_files = [
            ("Logistic Regression", cm_dir / "logistic_regression_confusion_matrix.png"),
            ("Naive Bayes", cm_dir / "multinomial_naive_bayes_confusion_matrix.png"),
            ("Linear SVM", cm_dir / "linear_svm_confusion_matrix.png"),
        ]

        for i, (m_label, img_p) in enumerate(cm_files):
            col_target = [col1, col2, col3][i]
            with col_target:
                st.markdown(f"**{m_label}**")
                if img_p.exists():
                    st.image(str(img_p), use_container_width=True)
                else:
                    st.caption("Figure unavailable")

    # -------------------------------------------------------------------------
    # TAB 4: Error Analysis
    # -------------------------------------------------------------------------
    with tab_err:
        st.markdown("#### Test Set Error Analysis")
        if config.ERROR_ANALYSIS_CSV.exists():
            err_df = pd.read_csv(config.ERROR_ANALYSIS_CSV)
            err_models = ["All"] + sorted(err_df["model"].unique().tolist())
            sel_err_m = st.selectbox("Filter Errors by Model:", err_models, key="rep_err_model_filter")

            filtered_err = err_df if sel_err_m == "All" else err_df[err_df["model"] == sel_err_m]
            st.caption(f"Showing **{len(filtered_err)}** misclassified records.")
            st.dataframe(filtered_err, use_container_width=True)
        else:
            st.info("error_analysis.csv not found.")

    # -------------------------------------------------------------------------
    # TAB 5: Model Comparison
    # -------------------------------------------------------------------------
    with tab_comp:
        st.markdown("#### Comparative Metric Leaderboard")
        if config.MODEL_COMPARISON_CSV.exists():
            comp_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)
            st.dataframe(comp_df, use_container_width=True)
        elif config.EVALUATION_RESULTS_CSV.exists():
            st.dataframe(pd.read_csv(config.EVALUATION_RESULTS_CSV), use_container_width=True)
        else:
            st.info("Comparison data not found.")