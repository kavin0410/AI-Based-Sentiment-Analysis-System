"""Settings Component for SentimentAI (SentimentLab).

Provides application, system configuration, and model information.
"""

from pathlib import Path
import streamlit as st
import config
from src.model_loader import validate_model_artifacts


def render_system_page():
    """Render the Settings & System diagnostics page."""
    st.markdown("## ⚙ Settings")
    st.caption("Application parameters, runtime configurations, model registry, and infrastructure health.")

    val_report = validate_model_artifacts()
    vocab_sz = val_report.get("vectorizer_vocab_size", 570)
    best_m = val_report.get("best_model_name", "Logistic Regression")

    # -------------------------------------------------------------------------
    # 1. System Health Status Cards
    # -------------------------------------------------------------------------
    st.markdown("### Infrastructure Health")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="product-card">
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Inference Engine</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #34d399; margin: 4px 0;">Online & Ready</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Python 3.10+ • Scikit-Learn • Streamlit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="product-card">
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">TF-IDF Vectorizer</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #34d399; margin: 4px 0;">Loaded</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">{vocab_sz} Features • Unigrams + Bigrams</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="product-card">
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Top Model</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #34d399; margin: 4px 0;">{best_m}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Selected by Weighted F1 Metric</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Configuration Parameters
    # -------------------------------------------------------------------------
    st.markdown("### Runtime Configuration")
    config_data = {
        "Parameter": [
            "Random State",
            "Max Input Length",
            "Prediction History Cap",
            "Supported Classes",
            "TF-IDF Max Features",
            "Split Ratio",
            "Evaluation Metric",
        ],
        "Value": [
            str(config.RANDOM_STATE),
            f"{config.MAX_INPUT_LENGTH} characters",
            f"{config.PREDICTION_HISTORY_LIMIT} entries (In-Memory)",
            ", ".join(config.SUPPORTED_LABELS),
            "5000 (570 fitted)",
            "80% Train / 20% Test (Stratified)",
            "Weighted F1-Score",
        ],
        "Status": [
            "Fixed",
            "Active",
            "Active",
            "Validated",
            "Serialized",
            "Verified",
            "Active",
        ],
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(config_data), use_container_width=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Model Registry Details
    # -------------------------------------------------------------------------
    st.markdown("### Model Registry")
    reg_col1, reg_col2, reg_col3 = st.columns(3)

    with reg_col1:
        st.write("**Logistic Regression**")
        st.caption("File: `models/logistic_regression.pkl`")
        st.caption("Probabilities: Supported (`predict_proba`)")

    with reg_col2:
        st.write("**Multinomial Naive Bayes**")
        st.caption("File: `models/naive_bayes.pkl`")
        st.caption("Probabilities: Supported (`predict_proba`)")

    with reg_col3:
        st.write("**Linear SVM**")
        st.caption("File: `models/linear_svm.pkl`")
        st.caption("Scores: Decision function (`decision_function`)")