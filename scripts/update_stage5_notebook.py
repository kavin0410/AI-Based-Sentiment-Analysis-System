"""Script to update notebooks/sentiment_analysis.ipynb with Stage 5 sections and real execution outputs."""

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

    # We want to keep existing cells up to Stage 4 Conclusion
    cutoff_idx = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "## Stage 5: Real-Time Sentiment Inference & Prediction Pipeline (Upcoming)" in src or "Stage 5: Real-Time Sentiment Inference" in src:
            cutoff_idx = i
            break

    if cutoff_idx is None:
        cutoff_idx = len(cells)

    kept_cells = cells[:cutoff_idx]
    print(f"Kept {len(kept_cells)} existing cells (Stages 1-4).")

    # Setup execution environment namespace
    ns = {"__file__": str(NOTEBOOK_PATH)}
    setup_code = """
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import numpy as np
import config
from src.model_loader import load_best_model, load_vectorizer, validate_model_artifacts
from src.predict import predict_sentiment, predict_batch, format_prediction_result, PredictionHistoryManager
"""
    exec(setup_code, ns)

    stage5_cells = []
    exec_counter = 102  # Continue execution count after Stage 4

    def add_section(md_text: str, code_str: str):
        nonlocal exec_counter
        stage5_cells.append(make_md_cell(md_text))
        out = run_code_capture_stdout(code_str, ns)
        stage5_cells.append(make_code_cell(code_str, out, exec_counter))
        exec_counter += 1

    # Stage 5 Header
    stage5_cells.append(make_md_cell(
        "---\n# Stage 5: Real-Time Sentiment Prediction Engine\n\n"
        "In this stage, we implement the real-time sentiment prediction engine. "
        "The engine dynamically loads the best model evaluated in Stage 4 (`results/best_model.json`) "
        "and uses the fitted TF-IDF vectorizer artifact (`models/tfidf_vectorizer.pkl`) to classify new, unseen text.\n"
        "**Zero Data Leakage:** No model retraining or vectorizer refitting occurs during inference."
    ))

    # 1. Validate Artifacts & Load Best Model
    add_section(
        "## 1. Validate Artifacts & Load Best Model Dynamically\n\n"
        "Read `results/best_model.json` to dynamically load the top-performing model without hardcoding.",
        """# Validate artifacts and load best model
validation_report = validate_model_artifacts()
print(f"Artifact Validation Status: {'VALID' if validation_report['valid'] else 'INVALID'}")

model, model_name, metadata = load_best_model()
print(f"Dynamically Loaded Best Model : {model_name}")
print(f"Stage 4 F1 Score Metric      : {metadata.get('f1_score', 0.0)*100:.2f}%")"""
    )

    # 2. Load TF-IDF Vectorizer
    add_section(
        "## 2. Load Pre-fitted TF-IDF Vectorizer\n\n"
        "Load the TF-IDF vectorizer artifact fitted on training data in Stage 3.",
        """# Load TF-IDF Vectorizer
vectorizer = load_vectorizer()
print(f"Loaded Vectorizer Vocabulary Size: {len(vectorizer.get_feature_names_out())} terms")"""
    )

    # 3. Test Real-Time Sentiment Prediction Pipeline
    add_section(
        "## 3. Real-Time Sentiment Prediction Pipeline Execution\n\n"
        "Pass a new, unseen text string through the full end-to-end prediction pipeline.",
        """# Single text prediction
sample_text = "I absolutely love this product! It exceeded all my expectations."
result = predict_sentiment(sample_text, model=model, vectorizer=vectorizer, model_name=model_name)

print("=== Prediction Output ===")
print(format_prediction_result(result))
print("\\n=== Class Probability / Score Breakdown ===")
print(result["probabilities"])"""
    )

    # 4. Test Multiple Representative Inputs
    add_section(
        "## 4. Test Multiple Sentiments & Negations\n\n"
        "Test positive, negative, neutral, and complex negation sentences.",
        """# Batch testing on varied sentences
test_sentences = [
    "I absolutely love this amazing product!",
    "This was the worst experience I have ever had.",
    "The package was delivered yesterday as scheduled.",
    "I didn't think the movie was bad at all.",
    "Customer support was quick, friendly, and resolved my issue."
]

results = predict_batch(test_sentences, model=model, vectorizer=vectorizer, model_name=model_name)

print(f"{'Input Text':<55} | {'Sentiment':<10} | {'Score/Confidence':<15}")
print("-" * 88)
for res in results:
    score_str = f"{res['score']*100:.1f}%" if res['score_type'] == 'probability' else f"{res['score']:.4f}"
    text_preview = res['text'][:52] + "..." if len(res['text']) > 55 else res['text']
    print(f"{text_preview:<55} | {res['sentiment']:<10} | {score_str:<15}")"""
    )

    # 5. Session History & CSV Export Test
    add_section(
        "## 5. In-Memory Session History & CSV Export\n\n"
        "Verify history tracking and export capabilities.",
        """# Test PredictionHistoryManager
history_mgr = PredictionHistoryManager(max_limit=50)
for res in results:
    history_mgr.add_prediction(res)

history_df = history_mgr.to_dataframe()
print(f"Recorded History Entries: {len(history_df)}")
print(history_df[["timestamp", "text", "sentiment", "score", "score_type"]].to_string(index=False))

csv_output = history_mgr.to_csv()
print(f"\\nExported CSV Preview (First 200 chars):\\n{csv_output[:200]}...")"""
    )

    # 6. Stage 5 Summary
    stage5_cells.append(make_md_cell(
        "## 6. Stage 5 Conclusion & Summary\n\n"
        "### Key Accomplishments in Stage 5:\n"
        "1. **Dynamic Model Resolution:** Implemented `src/model_loader.py` to dynamically load the best model (`results/best_model.json`).\n"
        "2. **End-to-End Inference Engine:** Implemented `src/predict.py` reusing Stage 2 NLP preprocessing (`preprocess_text`) and Stage 3 TF-IDF transformation (`transform_tfidf`).\n"
        "3. **Score & Probability Integrity:** Correctly extracted probabilities for probability models (`predict_proba`) and decision scores for SVM models (`decision_function`).\n"
        "4. **Input Validation:** Implemented strict input validation handling empty, whitespace-only, and excessively long inputs (>5000 chars).\n"
        "5. **Session History & CSV Export:** Created `PredictionHistoryManager` capping history at 50 records with instant CSV export.\n"
        "6. **Streamlit UI Extension:** Added Tab 1 (`⚡ Real-Time Predictor`) with live inference, result cards, preprocessing inspector, metadata view, and session history table.\n\n"
        "**Next Step:** Stage 6 - Production Sentiment Intelligence Dashboard."
    ))

    # Reassemble notebook
    all_cells = kept_cells + stage5_cells
    nb["cells"] = all_cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Notebook successfully updated with Stage 5 sections! Total cells: {len(all_cells)}")


if __name__ == "__main__":
    main()
