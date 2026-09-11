"""Model Intelligence Component for SentimentLab — Premium Light UI.

ML laboratory:
1. Model Leaderboard & Benchmarking
2. Architecture Specification Cards
3. Confusion Matrix Visualizer
4. Per-Class Performance Table
5. Model Playground
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

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_model_lab_page():
    """Render the Model Intelligence laboratory."""
    st.markdown(
        """
        <div class="sl-page-header">
            <div class="sl-page-eyebrow">ML LABORATORY</div>
            <h1 class="sl-page-title">Model Intelligence</h1>
            <p class="sl-page-subtitle">
                Benchmark, validate, and compare supervised ML classifiers on the held-out test split.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_intel, tab_playground = st.tabs(["Benchmarking & Metrics", "Model Playground"])

    # ------------------------------------------------------------------
    # TAB 1: Benchmarking
    # ------------------------------------------------------------------
    with tab_intel:
        best_meta = {}
        if config.BEST_MODEL_JSON.exists():
            with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
        best_name = best_meta.get("model_name", "Logistic Regression")

        # Leaderboard
        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Model Leaderboard</div>'
            f'<div class="sl-text-muted" style="margin-bottom:1rem;">'
            f'Top Model: <strong style="color:#6366F1;">{best_name}</strong> '
            f'| Weighted F1: <strong>{best_meta.get("f1_score", 0.3276)*100:.2f}%</strong></div>',
            unsafe_allow_html=True,
        )

        if config.MODEL_COMPARISON_CSV.exists():
            comp_df = pd.read_csv(config.MODEL_COMPARISON_CSV, index_col=0)
            st.dataframe(comp_df, use_container_width=True)
        elif config.EVALUATION_RESULTS_CSV.exists():
            st.dataframe(pd.read_csv(config.EVALUATION_RESULTS_CSV), use_container_width=True)

        # Plotly grouped bar for metrics
        if HAS_PLOTLY and config.EVALUATION_RESULTS_CSV.exists():
            try:
                eval_df = pd.read_csv(config.EVALUATION_RESULTS_CSV)
                model_col = next((c for c in eval_df.columns if c.lower() == "model"), None)
                
                # Check metrics columns
                metric_mapping = {
                    "accuracy": "Accuracy",
                    "precision_weighted": "Precision",
                    "precision": "Precision",
                    "recall_weighted": "Recall",
                    "recall": "Recall",
                    "f1_weighted": "F1 Score",
                    "f1_score": "F1 Score",
                }
                
                avail_metrics = [c for c in eval_df.columns if c.lower() in metric_mapping]
                
                if avail_metrics and model_col:
                    fig_g = go.Figure()
                    colors = ["#6366F1", "#8B5CF6", "#A855F7", "#06B6D4"]
                    for idx, mc in enumerate(avail_metrics):
                        vals = eval_df[mc].values
                        vals_pct = [float(v) * 100 if float(v) <= 1.0 else float(v) for v in vals]
                        fig_g.add_trace(go.Bar(
                            name=metric_mapping.get(mc.lower(), mc.title()),
                            x=eval_df[model_col].values,
                            y=vals_pct,
                            marker_color=colors[idx % len(colors)],
                            text=[f"{v:.1f}%" for v in vals_pct],
                            textposition="outside",
                            textfont=dict(family="Inter", size=10, color="#334155"),
                        ))
                    fig_g.update_layout(
                        barmode="group",
                        xaxis=dict(tickfont=dict(family="Inter", size=11, color="#64748B"),
                                   gridcolor="rgba(0,0,0,0)"),
                        yaxis=dict(ticksuffix="%", tickformat=".0f",
                                   range=[0, 60],
                                   gridcolor="rgba(99,102,241,0.08)",
                                   tickfont=dict(family="Inter", size=11, color="#94A3B8")),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=20, b=10, l=10, r=10),
                        height=260,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="center", x=0.5,
                                    font=dict(family="Inter", size=11)),
                    )
                    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                pass

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sl-spacer-md'></div>", unsafe_allow_html=True)

        # Architecture Specs
        st.markdown(
            '<div class="sl-section-label">Model Architecture</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        models_spec = [
            (col1, "Logistic Regression", best_name == "Logistic Regression",
             "lbfgs", "C=1.0", "1000", "Softmax"),
            (col2, "Multinomial Naive Bayes", best_name == "Multinomial Naive Bayes",
             "Multinomial", "alpha=1.0", "< 0.01s", "Log-Likelihood"),
            (col3, "Linear SVM", best_name == "Linear SVM",
             "LinearSVC", "C=1.0, dual=auto", "2000", "Decision Score"),
        ]

        for col, m_name, is_top, solver, reg, iters, prob_out in models_spec:
            with col:
                badge = "BEST MODEL" if is_top else "BENCHMARKED"
                badge_cls = "sl-badge-positive" if is_top else "sl-badge-neutral"
                card_cls = "sl-card-green" if is_top else ""
                st.markdown(
                    f"""
                    <div class="sl-card {card_cls}">
                        <div style="margin-bottom:0.5rem;">
                            <span class="sl-badge {badge_cls}" style="font-size:0.68rem;">{badge}</span>
                        </div>
                        <div style="font-size:1.15rem;font-weight:700;color:#0F172A;margin-bottom:0.75rem;">
                            {m_name}
                        </div>
                        <div style="font-size:0.82rem;color:#64748B;line-height:1.9;">
                            Algorithm: <code style="background:rgba(99,102,241,0.08);padding:2px 6px;border-radius:4px;">{solver}</code><br/>
                            Parameter: <code style="background:rgba(99,102,241,0.08);padding:2px 6px;border-radius:4px;">{reg}</code><br/>
                            Training: <code style="background:rgba(99,102,241,0.08);padding:2px 6px;border-radius:4px;">{iters}</code><br/>
                            Output: <strong style="color:#6366F1;">{prob_out}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='sl-spacer-md'></div>", unsafe_allow_html=True)

        # Confusion Matrices & Per-Class Metrics
        st.markdown(
            '<div class="sl-section-label">Confusion Matrix & Per-Class Performance</div>',
            unsafe_allow_html=True,
        )

        sel_matrix = st.selectbox(
            "Select Model:",
            ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"],
            key="matrix_sel",
        )

        col_img, col_class = st.columns([1, 1], gap="medium")
        with col_img:
            st.markdown('<div class="sl-card">', unsafe_allow_html=True)
            cm_map = {
                "Logistic Regression": config.RESULTS_PATH / "confusion_matrices" / "logistic_regression_confusion_matrix.png",
                "Multinomial Naive Bayes": config.RESULTS_PATH / "confusion_matrices" / "multinomial_naive_bayes_confusion_matrix.png",
                "Linear SVM": config.RESULTS_PATH / "confusion_matrices" / "linear_svm_confusion_matrix.png",
            }
            img_p = cm_map.get(sel_matrix)
            if img_p and img_p.exists():
                st.image(str(img_p), use_container_width=True)
            else:
                st.info("Confusion matrix not found.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_class:
            st.markdown(
                f'<div class="sl-card">'
                f'<div class="sl-section-title">Per-Class: {sel_matrix}</div>',
                unsafe_allow_html=True,
            )
            if config.PER_CLASS_METRICS_CSV.exists():
                pc_df = pd.read_csv(config.PER_CLASS_METRICS_CSV)
                # Find the model column safely (case-insensitive)
                model_col = next((c for c in pc_df.columns if c.lower() == "model"), None)
                if model_col:
                    filtered_pc = pc_df[pc_df[model_col].astype(str).str.lower() == sel_matrix.lower()]
                    if not filtered_pc.empty:
                        st.dataframe(filtered_pc, use_container_width=True)
                    else:
                        st.dataframe(pc_df, use_container_width=True)
                else:
                    st.dataframe(pc_df, use_container_width=True)
            else:
                st.info("Per-class metrics not available.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # TAB 2: Model Playground
    # ------------------------------------------------------------------
    with tab_playground:
        st.markdown(
            '<div class="sl-card">'
            '<div class="sl-section-title">Model Playground</div>'
            '<div class="sl-text-muted" style="margin-bottom:1rem;">'
            'Test how each trained model architecture classifies identical inputs.</div>',
            unsafe_allow_html=True,
        )

        vectorizer = load_vectorizer()

        col_sel, _ = st.columns([2, 2])
        with col_sel:
            selected_choice = st.selectbox(
                "Choose Model:",
                ["Best Model", "Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"],
                key="playground_model",
            )

        play_text = st.text_area(
            "Input Text",
            value="I didn't expect the quality to be so stellar! Exceeded all expectations.",
            height=110,
            key="playground_input",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Run Analysis \u2192", type="primary", key="btn_play_run"):
            if not play_text.strip():
                st.warning("Please enter text to test.")
            else:
                if selected_choice == "Best Model":
                    p_model, p_name, _ = load_best_model()
                else:
                    p_path = MODEL_NAME_TO_FILE.get(selected_choice)
                    p_model = load_model(p_path)
                    p_name = selected_choice

                with st.spinner(f"Evaluating with {p_name}..."):
                    play_res = predict_sentiment(play_text, model=p_model,
                                                vectorizer=vectorizer, model_name=p_name)
                    st.session_state["playground_last_res"] = (play_res, p_name)

        if "playground_last_res" in st.session_state:
            play_res, p_name = st.session_state["playground_last_res"]
            p_sent = play_res["sentiment"]
            p_score = play_res["score"]
            p_type = play_res["score_type"]
            badge_map = {"Positive": "sl-badge-positive", "Negative": "sl-badge-negative",
                         "Neutral": "sl-badge-neutral"}

            st.markdown(
                f"""
                <div class="sl-card" style="margin-top:1rem;">
                    <div class="sl-text-label">Evaluated by: {p_name}</div>
                    <div style="margin:0.6rem 0;">
                        <span class="sl-badge {badge_map.get(p_sent,'sl-badge-neutral')} sl-badge-lg">{p_sent}</span>
                    </div>
                    <div style="font-size:0.88rem;color:#475569;">
                        Confidence / Decision Score: <strong style="color:#0F172A;font-size:1.15rem;">{p_score*100:.1f}%</strong>
                        <span style="color:#64748B;">({p_type})</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )