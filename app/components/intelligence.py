"""Intelligence Component for AI Sentiment Intelligence (Stage 6).

Provides unified analytics across 4 sub-views:
1. Sentiment Overview (Dataset distribution + Current session predictions)
2. Model Performance (Comparison metrics table & grouped bar chart)
3. Confusion Analysis (Interactive model heatmaps & numerical matrices)
4. Error Intelligence (Total errors, confusion pairs, misclassified records)
"""

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config


def _load_evaluation_data():
    """Load Stage 4 evaluation results from disk."""
    eval_json = None
    best_model = None
    comparison_df = None
    error_df = None
    per_class_df = None

    if config.EVALUATION_RESULTS_JSON.exists():
        with open(config.EVALUATION_RESULTS_JSON, "r", encoding="utf-8") as f:
            eval_json = json.load(f)

    if config.BEST_MODEL_JSON.exists():
        with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
            best_model = json.load(f)

    if config.MODEL_COMPARISON_CSV.exists():
        comparison_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)

    if config.ERROR_ANALYSIS_CSV.exists():
        error_df = pd.read_csv(config.ERROR_ANALYSIS_CSV)

    if config.PER_CLASS_METRICS_CSV.exists():
        per_class_df = pd.read_csv(config.PER_CLASS_METRICS_CSV)

    return eval_json, best_model, comparison_df, error_df, per_class_df


def render_intelligence_page():
    """Render the unified Sentiment Intelligence dashboard."""
    st.markdown("## 📊 Sentiment Intelligence Analytics")
    st.caption("Unified performance insights, exploratory data distributions, confusion matrices, and error intelligence.")

    eval_json, best_model, comparison_df, error_df, per_class_df = _load_evaluation_data()

    tab_overview, tab_perf, tab_cm, tab_err = st.tabs([
        "📈 Sentiment Overview",
        "🏆 Model Performance",
        "🧩 Confusion Analysis",
        "🔍 Error Intelligence",
    ])

    # -------------------------------------------------------------------------
    # SUB-TAB 1: Sentiment Overview
    # -------------------------------------------------------------------------
    with tab_overview:
        st.markdown("#### 📈 Corpus & Session Sentiment Overview")

        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.markdown("##### 1. Cleaned Dataset Class Breakdown")
            st.markdown("- 🟢 **Positive:** 20 samples (33.3%)")
            st.markdown("- 🔴 **Negative:** 20 samples (33.3%)")
            st.markdown("- ⚪ **Neutral:** 20 samples (33.3%)")

            if config.SENTIMENT_DIST_PLOT.exists():
                st.image(str(config.SENTIMENT_DIST_PLOT), use_container_width=True)

        with col_dist2:
            st.markdown("##### 2. Current Session Predictions Analytics")
            history_mgr = st.session_state.get("history_manager")
            if history_mgr and history_mgr.get_history():
                h_df = history_mgr.to_dataframe()
                s_counts = h_df["sentiment"].value_counts()
                total_session = len(h_df)

                s_pos = s_counts.get("Positive", 0)
                s_neg = s_counts.get("Negative", 0)
                s_neu = s_counts.get("Neutral", 0)

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Positive", f"{s_pos}", f"{s_pos/total_session*100:.1f}%")
                with col_m2:
                    st.metric("Negative", f"{s_neg}", f"{s_neg/total_session*100:.1f}%")
                with col_m3:
                    st.metric("Neutral", f"{s_neu}", f"{s_neu/total_session*100:.1f}%")
            else:
                st.info("No predictions made in current session yet. Analyze text in the ⚡ Analyze tab to view live session metrics.")

        st.divider()
        st.markdown("##### 3. Text Length & Word Count Distribution")
        if config.TEXT_LENGTH_PLOT.exists():
            st.image(str(config.TEXT_LENGTH_PLOT), use_container_width=True)

    # -------------------------------------------------------------------------
    # SUB-TAB 2: Model Performance
    # -------------------------------------------------------------------------
    with tab_perf:
        st.markdown("#### 🏆 Model Performance Comparison")

        if best_model:
            st.success(
                f"🏆 **Selected Best Model:** `{best_model['model_name']}` | "
                f"Weighted F1: **{best_model['f1_score']*100:.2f}%** | "
                f"Accuracy: **{best_model['accuracy']*100:.2f}%**"
            )

        if comparison_df is not None:
            st.markdown("##### All Model Evaluation Metrics Table")
            styled_df = comparison_df.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                if col in styled_df.columns:
                    styled_df[col] = styled_df[col].apply(lambda x: f"{x*100:.2f}%")
            st.dataframe(styled_df, use_container_width=True)

        if config.MODEL_PERFORMANCE_PLOT.exists():
            st.markdown("##### 📊 Metric Comparison Chart")
            st.image(str(config.MODEL_PERFORMANCE_PLOT), use_container_width=True)

    # -------------------------------------------------------------------------
    # SUB-TAB 3: Confusion Analysis
    # -------------------------------------------------------------------------
    with tab_cm:
        st.markdown("#### 🧩 Confusion Matrix Analysis")

        cm_dir = config.CONFUSION_MATRICES_DIR
        if cm_dir.exists():
            st.markdown("##### 🏆 Best Model Confusion Matrix")
            if config.BEST_MODEL_CM_PLOT.exists():
                st.image(str(config.BEST_MODEL_CM_PLOT), use_container_width=True)

            st.divider()
            st.markdown("##### All Models Heatmaps")
            cm_files = sorted(cm_dir.glob("*_confusion_matrix.png"))
            if cm_files:
                cols = st.columns(min(len(cm_files), 3))
                for i, cm_file in enumerate(cm_files):
                    with cols[i % 3]:
                        m_name = cm_file.stem.replace("_confusion_matrix", "").replace("_", " ").title()
                        st.markdown(f"**{m_name}**")
                        st.image(str(cm_file), use_container_width=True)

        if config.CONFUSION_MATRICES_JSON.exists():
            with st.expander("📋 Numerical Confusion Matrices (JSON)", expanded=False):
                with open(config.CONFUSION_MATRICES_JSON, "r", encoding="utf-8") as f:
                    cm_data = json.load(f)
                for m_name, data in cm_data.items():
                    st.markdown(f"**{m_name}**")
                    labels = data.get("labels", ["Negative", "Neutral", "Positive"])
                    cm_df = pd.DataFrame(data["matrix"], index=labels, columns=labels)
                    cm_df.index.name = "Actual"
                    cm_df.columns.name = "Predicted"
                    st.dataframe(cm_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # SUB-TAB 4: Error Intelligence
    # -------------------------------------------------------------------------
    with tab_err:
        st.markdown("#### 🔍 Error Intelligence & Misclassification Inspection")

        if error_df is None or error_df.empty:
            if error_df is not None and error_df.empty:
                st.success("🎉 No misclassifications found! All predictions were 100% correct.")
            else:
                st.warning("Error analysis artifact not found. Run `python run_stage4.py`.")
        else:
            total_errors = len(error_df)
            models_err = error_df["model"].nunique()
            st.markdown(f"**Total Test Errors:** `{total_errors}` across `{models_err}` models")

            model_filter = st.selectbox(
                "Filter by model:",
                ["All Models"] + sorted(error_df["model"].unique().tolist()),
                key="intel_error_filter",
            )

            disp_err = error_df[error_df["model"] == model_filter] if model_filter != "All Models" else error_df
            st.dataframe(
                disp_err[["model", "text", "actual_sentiment", "predicted_sentiment"]].reset_index(drop=True),
                use_container_width=True,
            )

            st.divider()
            st.markdown("##### 🔄 Common Confusion Pairs")
            if not disp_err.empty:
                confusion_pairs = (
                    disp_err.groupby(["actual_sentiment", "predicted_sentiment"])
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Count", ascending=False)
                )
                confusion_pairs.columns = ["Actual Class", "Predicted Class", "Error Count"]
                st.dataframe(confusion_pairs.reset_index(drop=True), use_container_width=True)
