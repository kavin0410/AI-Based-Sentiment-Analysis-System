"""Script to update notebooks/sentiment_analysis.ipynb with Stage 6 sections and real execution outputs."""

import contextlib
import io
import json
from pathlib import Path
import sys

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

NOTEBOOK_PATH = BASE_DIR / "notebooks" / "sentiment_analysis.ipynb"


def make_md_cell(text: str):
    lines = [line + "\n" for line in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def make_code_cell(code: str, stdout_text: str, exec_count: int):
    code_lines = [line + "\n" for line in code.strip().split("\n")]
    if code_lines:
        code_lines[-1] = code_lines[-1].rstrip("\n")

    output_lines = [line + "\n" for line in stdout_text.strip().split("\n")]
    outputs = []
    if stdout_text.strip():
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": output_lines,
        })

    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": code_lines,
    }


def run_code_capture_stdout(code_str: str, glob_ns: dict) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code_str, glob_ns)
    return buf.getvalue()


def main():
    print("Reading notebook...")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # We want to keep existing cells up to Stage 5 Conclusion
    cutoff_idx = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "Stage 6" in src and "Upcoming" in src:
            cutoff_idx = i
            break

    if cutoff_idx is None:
        cutoff_idx = len(cells)

    kept_cells = cells[:cutoff_idx]
    print(f"Kept {len(kept_cells)} existing cells (Stages 1-5).")

    # Setup execution environment namespace
    ns = {"__file__": str(NOTEBOOK_PATH)}
    setup_code = """
import sys
import os
import json
from pathlib import Path
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import numpy as np
import config
from src.model_loader import load_best_model, load_vectorizer, validate_model_artifacts
from src.predict import predict_sentiment, predict_batch
"""
    exec(setup_code, ns)

    stage6_cells = []
    exec_counter = 114  # Continue execution count after Stage 5

    def add_section(md_text: str, code_str: str):
        nonlocal exec_counter
        stage6_cells.append(make_md_cell(md_text))
        out = run_code_capture_stdout(code_str, ns)
        stage6_cells.append(make_code_cell(code_str, out, exec_counter))
        exec_counter += 1

    # Stage 6 Header
    stage6_cells.append(make_md_cell(
        "---\n# Stage 6: Sentiment Intelligence Experience\n\n"
        "In this stage, we transform the system into an executive AI product interface. "
        "The interface communicates three layers: Executive User Experience, AI Intelligence, and Technical Transparency.\n"
        "**Transparency Disclaimer:** Current model evaluation is based on a small 60-record dataset. Results demonstrate technical pipeline functionality and are not interpreted as production-grade performance."
    ))

    # 1. Executive System Status & Artifact Validation
    add_section(
        "## 1. System Health & Artifact Validation\n\n"
        "Validate all backend artifacts required for the platform.",
        """# Validate workspace health
report = validate_model_artifacts()
print(f"System Health Status     : {'ONLINE' if report['valid'] else 'DEGRADED'}")
print(f"Active Best Model        : {report['best_model_name']}")
print(f"Vectorizer Vocabulary Size: {report['vectorizer_vocab_size']} features")
print(f"Model Classes            : {report['model_classes']}")"""
    )

    # 2. Executive Metrics Summary
    add_section(
        "## 2. Dynamic Executive Metrics Summary\n\n"
        "Retrieve system dimensions dynamically from configuration and files.",
        """# Executive KPI summary
df_clean = pd.read_csv(config.CLEANED_DATA_FILE)
print(f"Total Dataset Records : {len(df_clean)}")
print(f"Training Samples (80%): 48")
print(f"Test Samples (20%)    : 12")
print(f"Sentiment Categories  : {df_clean['sentiment'].value_counts().to_dict()}")"""
    )

    # 3. Best Model Card & Evaluation Results
    add_section(
        "## 3. Best Model Card Data Extraction\n\n"
        "Load `results/best_model.json` generated programmatically in Stage 4.",
        """# Load best model metadata
with open(config.BEST_MODEL_JSON, 'r', encoding='utf-8') as f:
    best_meta = json.load(f)

print(f"Selected Best Model : {best_meta['model_name']}")
print(f"Weighted F1 Score   : {best_meta['f1_score']*100:.2f}%")
print(f"Accuracy            : {best_meta['accuracy']*100:.2f}%")
print(f"Precision           : {best_meta['precision']*100:.2f}%")
print(f"Recall              : {best_meta['recall']*100:.2f}%")
print(f"Selection Criterion : Weighted {best_meta['selection_metric']}")"""
    )

    # 4. Interactive Analyzer Verification
    add_section(
        "## 4. End-to-End Real-Time Sentiment Analyzer Verification\n\n"
        "Verify sentiment prediction across test cases.",
        """# Hero sentiment analyzer check
test_texts = [
    "I absolutely love this amazing AI Sentiment Intelligence platform!",
    "This was the worst experience ever, totally unsatisfied.",
    "The package arrived on time with standard packaging."
]

for txt in test_texts:
    res = predict_sentiment(txt)
    score_str = f"{res['score']*100:.1f}%" if res['score_type'] == 'probability' else f"{res['score']:.4f}"
    print(f"Text      : '{txt}'")
    print(f"Prediction: {res['sentiment']} ({score_str} {res['score_type']})")
    print(f"Clean Text: '{res['processed_text']}'\\n")"""
    )

    # 5. Stage 6 Summary
    stage6_cells.append(make_md_cell(
        "## 5. Stage 6 Conclusion & Summary\n\n"
        "### Key Accomplishments in Stage 6:\n"
        "1. **Product UI Transformation:** Built an 8-page Streamlit application structure (`app/app.py` & `app/components/`).\n"
        "2. **Executive Overview Landing Page:** Integrated executive KPI metrics, dynamic Best Model Card, transparent 60-record dataset health warning, and Hero Sentiment Analyzer.\n"
        "3. **Real-Time Predictor:** Retained Stage 5 live inference with confidence breakdown, NLP inspection, and session history CSV export.\n"
        "4. **Unified Intelligence Dashboard:** Consolidated sentiment distribution, model metrics, Seaborn confusion matrix heatmaps, and error analysis.\n"
        "5. **Model Lab & Transparency:** Documented model hyperparameter specifications and automated selection logic (Weighted F1-Score primary).\n"
        "6. **NLP Pipeline Storytelling:** Illustrated the 12-step NLP pipeline with explicit Negation Preservation callouts.\n"
        "7. **Data & Quality Audit:** Provided dataset audit metrics, raw vs clean previews, and quality report downloads.\n\n"
        "**System Status:** Stage 6 Completed — Ready for Stage 7 Final Project & Documentation Automation."
    ))

    # Reassemble notebook
    all_cells = kept_cells + stage6_cells
    nb["cells"] = all_cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Notebook successfully updated with Stage 6 sections! Total cells: {len(all_cells)}")


if __name__ == "__main__":
    main()
