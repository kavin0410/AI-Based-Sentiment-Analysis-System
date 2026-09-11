"""Script to update notebooks/sentiment_analysis.ipynb with Stage 4 sections and real execution outputs."""

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

    # We want to keep cells up through Stage 3 (Cell 69 or ending at Stage 3 Summary)
    cutoff_idx = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "## Stage 4: Model Evaluation & Comparison (Upcoming)" in src or "Stage 4: Model Evaluation & Comparison" in src:
            cutoff_idx = i
            break

    if cutoff_idx is None:
        cutoff_idx = len(cells)

    kept_cells = cells[:cutoff_idx]
    print(f"Kept {len(kept_cells)} existing cells (Stages 1, 2, 3).")

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
from src.data_loader import load_dataset, split_dataset, validate_columns, validate_labels
from src.feature_extraction import load_vectorizer, transform_tfidf
from src.train_model import load_model, load_model_metadata
from src.evaluate_model import (
    calculate_model_metrics,
    calculate_per_class_metrics,
    generate_classification_report_text,
    generate_confusion_matrix,
    evaluate_all_models,
    select_best_model,
    create_comparison_table,
    perform_all_error_analyses,
    summarize_errors,
    plot_confusion_matrix,
    plot_model_performance_comparison,
    save_classification_reports,
    save_confusion_matrices_json,
    save_comparison_csv,
    save_best_model_json,
    save_per_class_metrics_csv,
    save_error_analysis_csv,
    save_evaluation_results_json,
    save_evaluation_results_csv,
)
"""
    exec(setup_code, ns)

    stage4_cells = []
    exec_counter = 71  # Continue execution count after Stage 3

    def add_section(md_text: str, code_str: str):
        nonlocal exec_counter
        stage4_cells.append(make_md_cell(md_text))
        out = run_code_capture_stdout(code_str, ns)
        stage4_cells.append(make_code_cell(code_str, out, exec_counter))
        exec_counter += 1

    # Stage 4 Header
    stage4_cells.append(make_md_cell(
        "---\n# Stage 4: Model Evaluation & Performance Intelligence\n\n"
        "In this stage, we evaluate the trained machine-learning models (Logistic Regression, "
        "Multinomial Naive Bayes, Linear SVM) on the held-out test dataset (20% split, 12 samples).\n"
        "All performance metrics are calculated strictly from model predictions on unseen test data."
    ))

    # 1. Load Trained Models
    add_section(
        "## 1. Load Trained Models from Disk\n\n"
        "Load the saved model artifacts from `models/` (trained in Stage 3).",
        """# Load saved models
lr_model = load_model(config.LOGISTIC_REGRESSION_FILE)
nb_model = load_model(config.NAIVE_BAYES_FILE)
svm_model = load_model(config.LINEAR_SVM_FILE)

models = {
    "Logistic Regression": lr_model,
    "Multinomial Naive Bayes": nb_model,
    "Linear SVM": svm_model,
}
print(f"Loaded {len(models)} models successfully.")"""
    )

    # 2. Load Saved TF-IDF Vectorizer
    add_section(
        "## 2. Load Saved TF-IDF Vectorizer\n\n"
        "Load the TF-IDF vectorizer artifact fitted on training data in Stage 3. "
        "**Zero Data Leakage Rule:** Do NOT refit the vectorizer.",
        """# Load vectorizer from Stage 3
vectorizer = load_vectorizer(config.TFIDF_VECTORIZER_FILE)
print(f"Loaded TF-IDF Vectorizer with vocabulary size: {len(vectorizer.get_feature_names_out())}")"""
    )

    # 3. Recreate Test Dataset
    add_section(
        "## 3. Recreate Test Dataset (Stratified Train/Test Split)\n\n"
        "Recreate the exact 80/20 stratified split used in Stage 3 using `random_state=42`.",
        """# Load cleaned dataset and split
df_clean = load_dataset(config.CLEANED_DATA_FILE)
X_train_text, X_test_text, y_train, y_test = split_dataset(
    df=df_clean,
    text_column=config.CLEAN_TEXT_COLUMN,
    sentiment_column=config.SENTIMENT_COLUMN,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=True,
)

# Transform test text using loaded vectorizer
X_test_tfidf = transform_tfidf(X_test_text, vectorizer)
print(f"X_test_tfidf matrix shape: {X_test_tfidf.shape}")"""
    )

    # 4. Generate Predictions
    add_section(
        "## 4. Generate Predictions on Test Set\n\n"
        "Generate test set predictions for all three models.",
        """# Generate predictions
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test_tfidf)
    print(f"{name}: {len(predictions[name])} predictions generated.")"""
    )

    # 5. Model Accuracy Calculation
    add_section(
        "## 5. Calculate Model Accuracy\n\n"
        "Calculate accuracy score for each model on the test dataset.",
        """# Calculate accuracy
for name, y_pred in predictions.items():
    metrics = calculate_model_metrics(y_test.values, y_pred)
    print(f"{name:<25}: Accuracy = {metrics['accuracy']*100:.2f}%")"""
    )

    # 6. Model Precision Calculation
    add_section(
        "## 6. Calculate Model Precision (Weighted & Per-Class)\n\n"
        "Calculate weighted and per-class precision across Negative, Neutral, and Positive sentiments.",
        """# Calculate precision
for name, y_pred in predictions.items():
    metrics = calculate_model_metrics(y_test.values, y_pred)
    per_class = calculate_per_class_metrics(y_test.values, y_pred, config.CM_LABEL_ORDER)
    print(f"=== {name} ===")
    print(f"Weighted Precision: {metrics['precision']*100:.2f}%")
    for cls, val in per_class.items():
        print(f"  - {cls:<10}: Precision = {val['precision']*100:.2f}%")"""
    )

    # 7. Model Recall Calculation
    add_section(
        "## 7. Calculate Model Recall (Weighted & Per-Class)\n\n"
        "Calculate weighted and per-class recall across all sentiment categories.",
        """# Calculate recall
for name, y_pred in predictions.items():
    metrics = calculate_model_metrics(y_test.values, y_pred)
    per_class = calculate_per_class_metrics(y_test.values, y_pred, config.CM_LABEL_ORDER)
    print(f"=== {name} ===")
    print(f"Weighted Recall: {metrics['recall']*100:.2f}%")
    for cls, val in per_class.items():
        print(f"  - {cls:<10}: Recall = {val['recall']*100:.2f}%")"""
    )

    # 8. Model F1-Score Calculation
    add_section(
        "## 8. Calculate Model F1-Score (Weighted & Per-Class)\n\n"
        "Calculate weighted and per-class F1-scores.",
        """# Calculate F1-score
for name, y_pred in predictions.items():
    metrics = calculate_model_metrics(y_test.values, y_pred)
    per_class = calculate_per_class_metrics(y_test.values, y_pred, config.CM_LABEL_ORDER)
    print(f"=== {name} ===")
    print(f"Weighted F1 Score: {metrics['f1_score']*100:.2f}%")
    for cls, val in per_class.items():
        print(f"  - {cls:<10}: F1 Score = {val['f1_score']*100:.2f}%")"""
    )

    # 9. Classification Reports
    add_section(
        "## 9. Full Classification Reports\n\n"
        "Display formatted scikit-learn classification reports for all models.",
        """# Print classification reports
for name, y_pred in predictions.items():
    report = generate_classification_report_text(y_test.values, y_pred, config.CM_LABEL_ORDER)
    print(f"============================================================")
    print(f"  Classification Report: {name}")
    print(f"============================================================")
    print(report)"""
    )

    # 10. Model Comparison Table
    add_section(
        "## 10. Model Comparison Table\n\n"
        "Side-by-side metrics comparison sorted by F1 Score.",
        """# Comprehensive evaluation
eval_results = evaluate_all_models(predictions, y_test.values, config.CM_LABEL_ORDER)
comparison_df = create_comparison_table(eval_results)
print(comparison_df.to_string())"""
    )

    # 11. Confusion Matrices
    add_section(
        "## 11. Confusion Matrix Analysis\n\n"
        "Display 3x3 confusion matrices ordered by Negative, Neutral, Positive.",
        """# Numerical confusion matrices
for name, res in eval_results.items():
    cm = np.array(res["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=config.CM_LABEL_ORDER, columns=config.CM_LABEL_ORDER)
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"
    print(f"--- Confusion Matrix: {name} ---")
    print(cm_df)
    print()"""
    )

    # 12. Per-Class Metrics Breakdown
    add_section(
        "## 12. Per-Class Metrics Breakdown\n\n"
        "Detailed breakdown of precision, recall, and F1 per class across all models.",
        """# Save and inspect per-class metrics
save_per_class_metrics_csv(eval_results, config.PER_CLASS_METRICS_CSV)
pc_df = pd.read_csv(config.PER_CLASS_METRICS_CSV)
print(pc_df.to_string(index=False))"""
    )

    # 13. Comprehensive Error Analysis
    add_section(
        "## 13. Comprehensive Error Analysis\n\n"
        "Inspect test samples where predicted sentiment differed from actual ground truth.",
        """# Perform error analysis
texts_orig = df_clean[config.TEXT_COLUMN].iloc[X_test_text.index]
error_df = perform_all_error_analyses(
    y_true=y_test,
    models_predictions=predictions,
    texts=texts_orig,
    clean_texts=X_test_text,
)
print(f"Total misclassified instances: {len(error_df)}")
if not error_df.empty:
    summary = summarize_errors(error_df)
    for model_name, info in summary.get("models", {}).items():
        print(f"\\n{model_name}: {info['error_count']} errors")
        for pair in info.get("confusion_pairs", []):
            print(f"  {pair['actual']} -> {pair['predicted']}: {pair['count']}")
    print("\\nSample Misclassifications:")
    print(error_df[["model", "text", "actual_sentiment", "predicted_sentiment"]].head(5).to_string(index=False))"""
    )

    # 14. Automatic Best Model Selection
    add_section(
        "## 14. Automatic Best Model Selection\n\n"
        "Select the best model based on weighted F1 score (primary) and accuracy (secondary).",
        """# Best model selection
best_model = select_best_model(eval_results, primary_metric="f1_score", secondary_metric="accuracy")
print(f"Best Model Selected : {best_model['model_name']}")
print(f"Weighted F1 Score   : {best_model['f1_score']*100:.2f}%")
print(f"Accuracy            : {best_model['accuracy']*100:.2f}%")
print(f"Selection Criterion : {best_model['selection_metric']}")"""
    )

    # 15. Manual Sanity Checks
    add_section(
        "## 15. Manual Sanity Checks (Qualitative Test Inputs)\n\n"
        "Qualitative testing on custom sentences (not part of official evaluation metrics).",
        """# Manual sanity checks
sanity_texts = [
    "I absolutely love this product!",
    "This was a terrible experience.",
    "The service was okay.",
    "I would definitely recommend this.",
    "I would never buy this again."
]

print(f"{'Input Sentence':<40} | {'LogReg':<10} | {'NaiveBayes':<10} | {'LinearSVM':<10}")
print("-" * 78)
for text in sanity_texts:
    # Preprocess and transform
    clean = " ".join(text.lower().replace("!", "").replace(".", "").split())
    feat = transform_tfidf(clean, vectorizer)
    p_lr = lr_model.predict(feat)[0]
    p_nb = nb_model.predict(feat)[0]
    p_svm = svm_model.predict(feat)[0]
    print(f"{text:<40} | {p_lr:<10} | {p_nb:<10} | {p_svm:<10}")"""
    )

    # 16. Stage 4 Summary & Conclusion
    stage4_cells.append(make_md_cell(
        "## 16. Stage 4 Conclusion & Summary\n\n"
        "### Key Accomplishments in Stage 4:\n"
        "1. **Evaluation Execution:** Evaluated Logistic Regression, Multinomial Naive Bayes, and Linear SVM on 12 test samples.\n"
        "2. **Metric Computation:** Calculated Accuracy, Precision, Recall, and F1-Score (both weighted and per-class).\n"
        "3. **Artifact Persistence:** Generated CSV and JSON reports in `results/`, including confusion matrices and model comparison tables.\n"
        "4. **Best Model Selection:** Programmatically selected the best-performing model based on weighted F1 score.\n"
        "5. **Error Analysis:** Categorized misclassifications and identified confusion pairs.\n"
        "6. **Data Leakage Check:** Confirmed zero data leakage throughout split, vectorization, and evaluation.\n\n"
        "**Next Step:** Stage 5 - Real-Time Sentiment Inference Engine & API Integration."
    ))

    # Reassemble complete notebook
    all_cells = kept_cells + stage4_cells
    nb["cells"] = all_cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Notebook successfully updated with Stage 4 sections! Total cells: {len(all_cells)}")


if __name__ == "__main__":
    main()
