"""Model Lab Component for AI Sentiment Intelligence.

Provides ML model comparison, leaderboard, model detail cards, model selector,
confusion matrix visualizer, and per-class metrics.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import config


def render_model_lab_page():
    """Render the Model Lab page."""
    st.markdown("## 🤖 Model Lab")
    st.caption("Benchmark, compare, and inspect the performance of trained machine learning classifiers on the held-out test split.")

    # -------------------------------------------------------------------------
    # 1. Dynamic Metric Loading
    # -------------------------------------------------------------------------
    best_meta = {}
    if config.BEST_MODEL_JSON.exists():
        with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
            best_meta = json.load(f)

    best_name = best_meta.get("model_name", "Logistic Regression")

    comparison_df = None
    if config.MODEL_COMPARISON_CSV.exists():
        comparison_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)

    # -------------------------------------------------------------------------
    # 2. Leaderboard
    # -------------------------------------------------------------------------
    st.markdown("### 🏆 Model Leaderboard")
    st.caption(f"Active Top Model: **{best_name}** (Ranked by Weighted F1-Score: {best_meta.get('f1_score', 0.3276)*100:.2f}%)")

    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)
    else:
        # Fallback table from results/evaluation_results.csv
        if config.EVALUATION_RESULTS_CSV.exists():
            eval_csv_df = pd.read_csv(config.EVALUATION_RESULTS_CSV)
            st.dataframe(eval_csv_df, use_container_width=True)

    st.markdown(
        """
        <div style="background: rgba(56, 139, 253, 0.08); border-left: 3px solid #388bfd; padding: 8px 12px; border-radius: 4px; font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem;">
            ℹ️ <strong>Evaluation Context:</strong> Evaluated on 12 held-out test samples (balanced 4 Positive, 4 Negative, 4 Neutral).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # 3. Model Architecture Detail Cards
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Model Architecture Specifications")
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        badge = "⭐ TOP PERFORMER" if best_name == "Logistic Regression" else "BENCHMARKED"
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-size: 0.8rem; color: #388bfd; font-weight: 600;">{badge}</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin: 4px 0;">Logistic Regression</div>
                <div style="font-size: 0.85rem; color: #8b949e;">Linear probability estimator with multinomial softmax</div>
                <hr style="border-color: rgba(255,255,255,0.06); margin: 8px 0;" />
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Solver: <code>lbfgs</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Regularization C: <code>1.0</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Max Iterations: <code>1000</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Probability Output: <strong>Yes</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        badge = "⭐ TOP PERFORMER" if best_name == "Multinomial Naive Bayes" else "BENCHMARKED"
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-size: 0.8rem; color: #8b949e; font-weight: 600;">{badge}</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin: 4px 0;">Multinomial Naive Bayes</div>
                <div style="font-size: 0.85rem; color: #8b949e;">Probabilistic classifier assuming feature independence</div>
                <hr style="border-color: rgba(255,255,255,0.06); margin: 8px 0;" />
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Smoothing: <code>alpha=1.0</code> (Laplace)</div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Prior: <code>Uniform / Learned</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Training Speed: <code>Instant (<0.01s)</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Probability Output: <strong>Yes</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        badge = "⭐ TOP PERFORMER" if best_name == "Linear SVM" else "BENCHMARKED"
        st.markdown(
            f"""
            <div class="asi-card">
                <div style="font-size: 0.8rem; color: #8b949e; font-weight: 600;">{badge}</div>
                <div style="font-size: 1.25rem; font-weight: 700; margin: 4px 0;">Linear SVM (LinearSVC)</div>
                <div style="font-size: 0.85rem; color: #8b949e;">Maximum-margin hyperplane in 570-dim feature space</div>
                <hr style="border-color: rgba(255,255,255,0.06); margin: 8px 0;" />
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Penalty: <code>L2 Regularized</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Loss: <code>Squared Hinge</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Margin Maximization: <code>Yes</code></div>
                <div style="font-size: 0.8rem; color: #c9d1d9;">• Output: <strong>Decision Score</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -------------------------------------------------------------------------
    # 4. Confusion Matrix & Per-Class Metrics
    # -------------------------------------------------------------------------
    st.markdown("### 🧩 Confusion Matrix & Per-Class Performance")

    model_choice = st.selectbox(
        "Select Model to Inspect:",
        ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"],
        key="model_lab_sel",
    )

    col_cm1, col_cm2 = st.columns([1, 1])

    with col_cm1:
        st.markdown(f"##### Confusion Matrix: {model_choice}")
        img_map = {
            "Logistic Regression": config.RESULTS_PATH / "confusion_matrices" / "logistic_regression_confusion_matrix.png",
            "Multinomial Naive Bayes": config.RESULTS_PATH / "confusion_matrices" / "multinomial_naive_bayes_confusion_matrix.png",
            "Linear SVM": config.RESULTS_PATH / "confusion_matrices" / "linear_svm_confusion_matrix.png",
        }
        cm_path = img_map.get(model_choice)
        if cm_path and cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        elif (config.RESULTS_PATH / "best_model_confusion_matrix.png").exists():
            st.image(str(config.RESULTS_PATH / "best_model_confusion_matrix.png"), use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    with col_cm2:
        st.markdown(f"##### Per-Class Metrics: {model_choice}")
        if config.PER_CLASS_METRICS_CSV.exists():
            per_class_df = pd.read_csv(config.PER_CLASS_METRICS_CSV)
            filtered_pc = per_class_df[per_class_df["model"] == model_choice]
            if not filtered_pc.empty:
                st.dataframe(filtered_pc[["class", "precision", "recall", "f1_score", "support"]], use_container_width=True)
            else:
                st.dataframe(per_class_df, use_container_width=True)
        else:
            st.info("Per-class metrics table not found.")