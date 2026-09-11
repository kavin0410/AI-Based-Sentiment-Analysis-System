"""Model Intelligence Component for SentimentAI (SentimentLab).

Professional ML model evaluation laboratory:
- Logistic Regression, Naive Bayes, Linear SVM
- Accuracy, Precision, Recall, F1 score
- Confusion matrices
- Model leaderboard
- Selected best model
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import config


def render_model_lab_page():
    """Render the Model Intelligence laboratory."""
    st.markdown("## ◉ Model Intelligence")
    st.caption("Benchmark, validate, and compare supervised machine learning classifiers on the held-out test split.")

    # -------------------------------------------------------------------------
    # 1. Model Leaderboard
    # -------------------------------------------------------------------------
    best_meta = {}
    if config.BEST_MODEL_JSON.exists():
        with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
            best_meta = json.load(f)

    best_name = best_meta.get("model_name", "Logistic Regression")

    st.markdown("### Model Leaderboard")
    st.caption(f"Active Top Model: **{best_name}** | Ranked by Weighted F1-Score")

    if config.MODEL_COMPARISON_CSV.exists():
        comp_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)
        st.dataframe(comp_df, use_container_width=True)
    elif config.EVALUATION_RESULTS_CSV.exists():
        eval_df = pd.read_csv(config.EVALUATION_RESULTS_CSV)
        st.dataframe(eval_df, use_container_width=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Model Architecture & Specifications
    # -------------------------------------------------------------------------
    st.markdown("### Architecture & Specifications")
    col1, col2, col3 = st.columns(3)

    with col1:
        is_best = (best_name == "Logistic Regression")
        badge = "ACTIVE (RANK 1)" if is_best else "BENCHMARKED"
        st.markdown(
            f"""
            <div class="product-card">
                <div style="font-size: 0.75rem; font-weight: 600; color: #6366f1; letter-spacing: 0.05em;">{badge}</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Logistic Regression</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Multinomial linear model with L2 regularization</div>
                <div style="margin-top: 12px; font-size: 0.8rem; color: #cbd5e1; line-height: 1.8;">
                    • Solver: <code>lbfgs</code><br/>
                    • Regularization C: <code>1.0</code><br/>
                    • Max Iterations: <code>1000</code><br/>
                    • Probabilistic Output: <strong>Yes</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        is_best = (best_name == "Multinomial Naive Bayes")
        badge = "ACTIVE (RANK 1)" if is_best else "BENCHMARKED"
        st.markdown(
            f"""
            <div class="product-card">
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.05em;">{badge}</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Multinomial Naive Bayes</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Generative classifier with conditional independence</div>
                <div style="margin-top: 12px; font-size: 0.8rem; color: #cbd5e1; line-height: 1.8;">
                    • Smoothing: <code>alpha=1.0</code> (Laplace)<br/>
                    • Vocabulary: <code>570 n-grams</code><br/>
                    • Training Latency: <code>< 0.01s</code><br/>
                    • Probabilistic Output: <strong>Yes</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        is_best = (best_name == "Linear SVM")
        badge = "ACTIVE (RANK 1)" if is_best else "BENCHMARKED"
        st.markdown(
            f"""
            <div class="product-card">
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.05em;">{badge}</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Linear SVM (LinearSVC)</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Maximum-margin hyperplane in sparse feature space</div>
                <div style="margin-top: 12px; font-size: 0.8rem; color: #cbd5e1; line-height: 1.8;">
                    • Penalty: <code>L2 Regularized</code><br/>
                    • Loss: <code>Squared Hinge</code><br/>
                    • Dual: <code>auto</code><br/>
                    • Probabilistic Output: <strong>Decision Score</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Confusion Matrices & Per-Class Details
    # -------------------------------------------------------------------------
    st.markdown("### Confusion Matrices & Per-Class Metrics")

    selected_m = st.selectbox(
        "Select Model to Inspect:",
        ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"],
        key="model_intel_sel",
    )

    col_cm, col_metrics = st.columns([1, 1])

    with col_cm:
        st.markdown(f"##### Confusion Matrix: {selected_m}")
        cm_map = {
            "Logistic Regression": config.RESULTS_PATH / "confusion_matrices" / "logistic_regression_confusion_matrix.png",
            "Multinomial Naive Bayes": config.RESULTS_PATH / "confusion_matrices" / "multinomial_naive_bayes_confusion_matrix.png",
            "Linear SVM": config.RESULTS_PATH / "confusion_matrices" / "linear_svm_confusion_matrix.png",
        }
        cm_img = cm_map.get(selected_m)
        if cm_img and cm_img.exists():
            st.image(str(cm_img), use_container_width=True)
        elif (config.RESULTS_PATH / "best_model_confusion_matrix.png").exists():
            st.image(str(config.RESULTS_PATH / "best_model_confusion_matrix.png"), use_container_width=True)
        else:
            st.info("Matrix figure not found.")

    with col_metrics:
        st.markdown(f"##### Per-Class Performance: {selected_m}")
        if config.PER_CLASS_METRICS_CSV.exists():
            pc_df = pd.read_csv(config.PER_CLASS_METRICS_CSV)
            m_pc = pc_df[pc_df["model"] == selected_m]
            if not m_pc.empty:
                st.dataframe(m_pc[["class", "precision", "recall", "f1_score", "support"]], use_container_width=True)
            else:
                st.dataframe(pc_df, use_container_width=True)
        else:
            st.info("Per-class metrics file not found.")