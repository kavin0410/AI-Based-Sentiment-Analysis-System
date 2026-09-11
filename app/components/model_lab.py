"""Model Intelligence & Playground Component for SentimentLab.

Professional ML model laboratory:
1. Leaderboard & Metrics:
   - Logistic Regression, Naive Bayes, Linear SVM
   - Accuracy, Precision, Recall, F1 Score
   - Dynamic best model identification from best_model.json
2. Performance Comparison & Visual Confusion Matrices
3. Model Playground:
   - Model Selector: Best Model, Logistic Regression, Naive Bayes, Linear SVM
   - Enter text and compare predictions between models
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import config
from src.model_loader import (
    load_best_model,
    load_model,
    load_vectorizer,
    MODEL_NAME_TO_FILE,
)
from src.predict import predict_sentiment


def render_model_lab_page():
    """Render the Model Intelligence laboratory & playground."""
    st.markdown("## ◉ Model Intelligence")
    st.caption("Benchmark, validate, and compare supervised machine learning classifiers on the held-out test split.")

    tab_intel, tab_playground = st.tabs(["Model Benchmarking & Matrices", "Model Playground"])

    # -------------------------------------------------------------------------
    # TAB 1: Benchmarking & Matrices
    # -------------------------------------------------------------------------
    with tab_intel:
        best_meta = {}
        if config.BEST_MODEL_JSON.exists():
            with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
        best_name = best_meta.get("model_name", "Logistic Regression")

        st.markdown("### Model Leaderboard")
        st.caption(f"Top Model: **{best_name}** | Ranked by Weighted F1-Score: **{best_meta.get('f1_score', 0.3276)*100:.2f}%**")

        if config.MODEL_COMPARISON_CSV.exists():
            comp_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)
            st.dataframe(comp_df, use_container_width=True)
        elif config.EVALUATION_RESULTS_CSV.exists():
            st.dataframe(pd.read_csv(config.EVALUATION_RESULTS_CSV), use_container_width=True)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Specifications Cards
        st.markdown("### Model Architecture Specifications")
        col1, col2, col3 = st.columns(3)

        models_spec = [
            (col1, "Logistic Regression", best_name == "Logistic Regression", "lbfgs", "C=1.0", "1000", "Yes (Softmax)"),
            (col2, "Multinomial Naive Bayes", best_name == "Multinomial Naive Bayes", "Multinomial", "alpha=1.0", "< 0.01s", "Yes (Log-Likelihood)"),
            (col3, "Linear SVM", best_name == "Linear SVM", "LinearSVC", "C=1.0, dual=auto", "2000", "Decision Score"),
        ]

        for col, m_name, is_top, solver, reg, iters, prob_out in models_spec:
            with col:
                badge = "⭐ RANK 1 (BEST)" if is_top else "BENCHMARKED"
                badge_color = "#38bdf8" if is_top else "#64748b"
                st.markdown(
                    f"""
                    <div class="product-card">
                        <div style="font-size: 0.72rem; font-weight: 700; color: {badge_color}; letter-spacing: 0.05em;">{badge}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin: 4px 0;">{m_name}</div>
                        <div style="margin-top: 10px; font-size: 0.82rem; color: #94a3b8; line-height: 1.8;">
                            • Algorithm: <code>{solver}</code><br/>
                            • Parameter: <code>{reg}</code><br/>
                            • Training: <code>{iters}</code><br/>
                            • Output: <strong style="color: #bae6fd;">{prob_out}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Confusion Matrices
        st.markdown("### Confusion Matrix Visualizer")
        sel_matrix_model = st.selectbox(
            "Select Model to Inspect Confusion Matrix:",
            ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"],
            key="matrix_model_sel",
        )

        col_img, col_class = st.columns([1, 1])
        with col_img:
            cm_map = {
                "Logistic Regression": config.RESULTS_PATH / "confusion_matrices" / "logistic_regression_confusion_matrix.png",
                "Multinomial Naive Bayes": config.RESULTS_PATH / "confusion_matrices" / "multinomial_naive_bayes_confusion_matrix.png",
                "Linear SVM": config.RESULTS_PATH / "confusion_matrices" / "linear_svm_confusion_matrix.png",
            }
            img_p = cm_map.get(sel_matrix_model)
            if img_p and img_p.exists():
                st.image(str(img_p), use_container_width=True)
            else:
                st.info("Confusion matrix heatmap not found.")

        with col_class:
            st.markdown(f"##### Per-Class Performance: {sel_matrix_model}")
            if config.PER_CLASS_METRICS_CSV.exists():
                pc_df = pd.read_csv(config.PER_CLASS_METRICS_CSV)
                filtered_pc = pc_df[pc_df["model"] == sel_matrix_model]
                if not filtered_pc.empty:
                    st.dataframe(filtered_pc[["class", "precision", "recall", "f1_score", "support"]], use_container_width=True)
                else:
                    st.dataframe(pc_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Model Playground
    # -------------------------------------------------------------------------
    with tab_playground:
        st.markdown("### Model Playground")
        st.caption("Interact directly with specific trained model artifacts to test how each architecture classifies identical inputs.")

        vectorizer = load_vectorizer()

        col_sel, _ = st.columns([2, 2])
        with col_sel:
            selected_choice = st.selectbox(
                "Choose Model:",
                ["Best Model", "Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"],
                key="playground_model_choice",
            )

        play_text = st.text_area(
            "Input Text for Model Playground:",
            value="I didn't expect the quality to be so stellar! Exceeded all expectations.",
            height=110,
            key="playground_input_area",
        )

        if st.button("Run Playground Analysis", type="primary", key="btn_play_run"):
            if not play_text.strip():
                st.warning("Please enter some text to test.")
            else:
                # Load selected model
                if selected_choice == "Best Model":
                    p_model, p_name, _ = load_best_model()
                else:
                    p_path = MODEL_NAME_TO_FILE.get(selected_choice)
                    p_model = load_model(p_path)
                    p_name = selected_choice

                with st.spinner(f"Evaluating with {p_name}..."):
                    play_res = predict_sentiment(play_text, model=p_model, vectorizer=vectorizer, model_name=p_name)

                p_sent = play_res["sentiment"]
                p_score = play_res["score"]
                p_type = play_res["score_type"]

                badge = "badge-positive" if p_sent == "Positive" else ("badge-negative" if p_sent == "Negative" else "badge-neutral")

                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="product-card">
                        <div style="font-size: 0.8rem; color: #7dd3fc; font-weight: 600;">EVALUATED BY: {p_name}</div>
                        <div style="margin: 0.5rem 0;">
                            <span class="sentiment-badge {badge}">{p_sent}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">
                            Score / Confidence: <strong style="color: #ffffff;">{p_score*100:.1f}%</strong> ({p_type})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )