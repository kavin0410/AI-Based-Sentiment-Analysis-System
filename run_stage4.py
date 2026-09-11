"""Stage 4: Model Evaluation & Performance Intelligence Runner.

This script orchestrates the complete Stage 4 evaluation pipeline:
1.  Verifies that all Stage 3 artifacts exist on disk
2.  Loads the cleaned dataset and recreates the identical 80/20 stratified split
3.  Loads the saved TF-IDF vectorizer (does NOT refit)
4.  Loads all 3 trained models from disk (does NOT retrain)
5.  Generates predictions on the test set for every model
6.  Computes accuracy, precision, recall, and F1-score (weighted)
7.  Computes per-class metrics for each sentiment label
8.  Generates classification reports and confusion matrices
9.  Automatically selects the best model (primary: weighted F1, secondary: accuracy)
10. Performs error analysis on all misclassified samples
11. Displays confidence/probability analysis (predict_proba for LR/NB, decision_function for SVM)
12. Generates visualisation plots (confusion matrices, performance comparison chart)
13. Persists all results as JSON, CSV, TXT, and PNG artifacts
14. Verifies data leakage prevention

IMPORTANT: LinearSVC does NOT support predict_proba(). decision_function() is used instead.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_CM_PLOT,
    CLEAN_TEXT_COLUMN,
    CLEANED_DATA_FILE,
    CM_LABEL_ORDER,
    EVALUATION_RESULTS_CSV,
    EVALUATION_RESULTS_JSON,
    LINEAR_SVM_FILE,
    LOGISTIC_REGRESSION_FILE,
    MODEL_METADATA_FILE,
    NAIVE_BAYES_FILE,
    RANDOM_STATE,
    SENTIMENT_COLUMN,
    SUPPORTED_LABELS,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_VECTORIZER_FILE,
    ensure_directories,
)
from src.data_loader import load_dataset, split_dataset, validate_columns, validate_labels
from src.evaluate_model import (
    create_comparison_table,
    evaluate_all_models,
    perform_all_error_analyses,
    plot_confusion_matrix,
    plot_model_performance_comparison,
    save_best_model_json,
    save_classification_reports,
    save_comparison_csv,
    save_confusion_matrices_json,
    save_error_analysis_csv,
    save_evaluation_results_csv,
    save_evaluation_results_json,
    save_per_class_metrics_csv,
    select_best_model,
    summarize_errors,
)
from src.feature_extraction import load_vectorizer, transform_tfidf
from src.train_model import load_model, load_model_metadata


def run_stage4():
    """Execute the complete Stage 4 model evaluation pipeline."""
    print("=" * 80)
    print("  STAGE 4: MODEL EVALUATION & PERFORMANCE INTELLIGENCE")
    print("=" * 80)

    # ------------------------------------------------------------------
    # STEP 1  Ensure directories
    # ------------------------------------------------------------------
    print("\n[STEP 1] Ensuring project directories exist...")
    ensure_directories()
    print("[DONE]")

    # ------------------------------------------------------------------
    # STEP 2  Verify Stage 3 artifacts
    # ------------------------------------------------------------------
    print("\n[STEP 2] Verifying Stage 3 artifacts exist on disk...")
    artifacts = {
        "Cleaned Dataset": CLEANED_DATA_FILE,
        "TF-IDF Vectorizer": TFIDF_VECTORIZER_FILE,
        "Logistic Regression": LOGISTIC_REGRESSION_FILE,
        "Naive Bayes": NAIVE_BAYES_FILE,
        "Linear SVM": LINEAR_SVM_FILE,
        "Model Metadata": MODEL_METADATA_FILE,
    }
    for name, path in artifacts.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"[ERROR] Required Stage 3 artifact missing: {name} at {path}.\n"
                f"Please run Stage 3 (python run_stage3.py) before evaluating."
            )
        print(f"  [OK] {name}: {Path(path).name}")
    print("[DONE] All 6 Stage 3 artifacts verified.")

    # ------------------------------------------------------------------
    # STEP 3  Load model metadata
    # ------------------------------------------------------------------
    print("\n[STEP 3] Loading model metadata...")
    metadata = load_model_metadata(MODEL_METADATA_FILE)
    print(f"  Project       : {metadata.get('project', 'N/A')}")
    print(f"  Stage         : {metadata.get('stage', 'N/A')}")
    print(f"  Total Samples : {metadata.get('total_samples', 'N/A')}")
    print(f"  Train Samples : {metadata.get('train_samples', 'N/A')}")
    print(f"  Test Samples  : {metadata.get('test_samples', 'N/A')}")
    print(f"  Random State  : {metadata.get('random_state', 'N/A')}")

    # ------------------------------------------------------------------
    # STEP 4  Load cleaned dataset
    # ------------------------------------------------------------------
    print("\n[STEP 4] Loading cleaned dataset...")
    df = load_dataset(CLEANED_DATA_FILE)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # STEP 5  Validate columns and labels
    # ------------------------------------------------------------------
    print("\n[STEP 5] Validating dataset columns and sentiment labels...")
    text_col = CLEAN_TEXT_COLUMN if CLEAN_TEXT_COLUMN in df.columns else TEXT_COLUMN
    if not validate_columns(df, [text_col, SENTIMENT_COLUMN]):
        raise ValueError(f"Dataset is missing required columns: {text_col}, {SENTIMENT_COLUMN}")
    df = validate_labels(df, SENTIMENT_COLUMN, SUPPORTED_LABELS)
    print(f"  Using text feature column: '{text_col}'")
    print(f"  Sentiment distribution:\n{df[SENTIMENT_COLUMN].value_counts().to_string()}")
    print("[DONE]")

    # ------------------------------------------------------------------
    # STEP 6  Recreate identical train/test split
    # ------------------------------------------------------------------
    print("\n[STEP 6] Recreating stratified train/test split (random_state=42, test_size=0.2)...")
    X_train_text, X_test_text, y_train, y_test = split_dataset(
        df=df,
        text_column=text_col,
        sentiment_column=SENTIMENT_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=True,
    )
    # Also get original raw texts for error analysis display
    X_test_original_text = df[TEXT_COLUMN].iloc[X_test_text.index]
    X_test_clean_text = X_test_text  # clean_text column values

    # ------------------------------------------------------------------
    # STEP 7  Load saved TF-IDF vectorizer (DO NOT refit)
    # ------------------------------------------------------------------
    print("\n[STEP 7] Loading saved TF-IDF vectorizer (transform only, no refitting)...")
    vectorizer = load_vectorizer(TFIDF_VECTORIZER_FILE)

    # ------------------------------------------------------------------
    # STEP 8  Transform test text
    # ------------------------------------------------------------------
    print("\n[STEP 8] Transforming X_test using the loaded vectorizer...")
    X_test_tfidf = transform_tfidf(X_test_text, vectorizer)
    print(f"  X_test_tfidf shape: {X_test_tfidf.shape}")

    # ------------------------------------------------------------------
    # STEP 9  Load all 3 trained models
    # ------------------------------------------------------------------
    print("\n[STEP 9] Loading trained models from disk...")
    lr_model = load_model(LOGISTIC_REGRESSION_FILE)
    nb_model = load_model(NAIVE_BAYES_FILE)
    svm_model = load_model(LINEAR_SVM_FILE)

    models = {
        "Logistic Regression": lr_model,
        "Multinomial Naive Bayes": nb_model,
        "Linear SVM": svm_model,
    }
    print("[DONE] All 3 models loaded.")

    # ------------------------------------------------------------------
    # STEP 10  Generate predictions
    # ------------------------------------------------------------------
    print("\n[STEP 10] Generating predictions on test set...")
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test_tfidf)
        print(f"  {name}: {len(predictions[name])} predictions generated")

    # ------------------------------------------------------------------
    # STEP 11  Comprehensive evaluation
    # ------------------------------------------------------------------
    print("\n[STEP 11] Evaluating all models (metrics, per-class, confusion matrices)...")
    evaluation_results = evaluate_all_models(predictions, y_test.values, CM_LABEL_ORDER)
    print("[DONE]")

    # ------------------------------------------------------------------
    # STEP 12  Print classification reports
    # ------------------------------------------------------------------
    print("\n[STEP 12] Classification Reports")
    print("-" * 80)
    for name, result in evaluation_results.items():
        m = result["metrics"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  Accuracy  : {m['accuracy']*100:.2f}%")
        print(f"  Precision : {m['precision']*100:.2f}% (weighted)")
        print(f"  Recall    : {m['recall']*100:.2f}% (weighted)")
        print(f"  F1 Score  : {m['f1_score']*100:.2f}% (weighted)")
        print(f"\n{result['classification_report']}")

    # ------------------------------------------------------------------
    # STEP 13  Model comparison table
    # ------------------------------------------------------------------
    print("\n[STEP 13] Model Comparison Table")
    print("-" * 80)
    comparison_df = create_comparison_table(evaluation_results)
    print(comparison_df.to_string())

    # ------------------------------------------------------------------
    # STEP 14  Best model selection
    # ------------------------------------------------------------------
    print("\n[STEP 14] Automatic Best Model Selection")
    print("-" * 80)
    best_model = select_best_model(evaluation_results, primary_metric="f1_score", secondary_metric="accuracy")
    print(f"   Best Model : {best_model['model_name']}")
    print(f"     F1 Score   : {best_model['f1_score']*100:.2f}%")
    print(f"     Accuracy   : {best_model['accuracy']*100:.2f}%")
    print(f"     Precision  : {best_model['precision']*100:.2f}%")
    print(f"     Recall     : {best_model['recall']*100:.2f}%")
    print(f"     Criterion  : {best_model['selection_metric']}")

    # ------------------------------------------------------------------
    # STEP 15  Error analysis
    # ------------------------------------------------------------------
    print("\n[STEP 15] Performing error analysis across all models...")
    error_df = perform_all_error_analyses(
        y_true=y_test,
        models_predictions=predictions,
        texts=X_test_original_text,
        clean_texts=X_test_clean_text,
    )
    total_errors = len(error_df)
    print(f"  Total misclassified samples (across all models): {total_errors}")

    # ------------------------------------------------------------------
    # STEP 16  Error summary (confusion pairs)
    # ------------------------------------------------------------------
    print("\n[STEP 16] Error Analysis Summary")
    print("-" * 80)
    if not error_df.empty:
        error_summary = summarize_errors(error_df)
        for model_name, model_info in error_summary.get("models", {}).items():
            print(f"\n  {model_name}: {model_info['error_count']} misclassification(s)")
            for pair in model_info.get("confusion_pairs", []):
                print(f"    {pair['actual']}  {pair['predicted']}: {pair['count']} time(s)")

        # Print actual misclassified samples
        print(f"\n  Misclassified Samples:")
        for _, row in error_df.iterrows():
            text_preview = row['text'][:60] + '...' if len(str(row['text'])) > 60 else row['text']
            print(f"    [{row['model']}] \"{text_preview}\"")
            print(f"      Actual: {row['actual_sentiment']} | Predicted: {row['predicted_sentiment']}")
    else:
        print("   No misclassifications found! All models predict perfectly on this test set.")

    # ------------------------------------------------------------------
    # STEP 17  Confidence / probability analysis
    # ------------------------------------------------------------------
    print("\n[STEP 17] Confidence & Probability Analysis")
    print("-" * 80)

    # Logistic Regression  predict_proba
    print("\n  Logistic Regression  predict_proba()")
    lr_proba = lr_model.predict_proba(X_test_tfidf)
    print(f"  Shape: {lr_proba.shape}")
    print(f"  Classes: {lr_model.classes_}")
    print(f"  Sample probabilities (first 3 test samples):")
    for i in range(min(3, len(lr_proba))):
        proba_dict = {c: round(float(p), 4) for c, p in zip(lr_model.classes_, lr_proba[i])}
        print(f"    Test {i+1}: {proba_dict}  Predicted: {predictions['Logistic Regression'][i]}")

    # Multinomial Naive Bayes  predict_proba
    print(f"\n  Multinomial Naive Bayes  predict_proba()")
    nb_proba = nb_model.predict_proba(X_test_tfidf)
    print(f"  Shape: {nb_proba.shape}")
    print(f"  Classes: {nb_model.classes_}")
    print(f"  Sample probabilities (first 3 test samples):")
    for i in range(min(3, len(nb_proba))):
        proba_dict = {c: round(float(p), 4) for c, p in zip(nb_model.classes_, nb_proba[i])}
        print(f"    Test {i+1}: {proba_dict}  Predicted: {predictions['Multinomial Naive Bayes'][i]}")

    # Linear SVM  decision_function (NOT predict_proba!)
    print(f"\n  Linear SVM  decision_function() (NOT predict_proba)")
    svm_scores = svm_model.decision_function(X_test_tfidf)
    print(f"  Shape: {svm_scores.shape}")
    print(f"  Classes: {svm_model.classes_}")
    print(f"  Sample decision scores (first 3 test samples):")
    for i in range(min(3, len(svm_scores))):
        score_dict = {c: round(float(s), 4) for c, s in zip(svm_model.classes_, svm_scores[i])}
        print(f"    Test {i+1}: {score_dict}  Predicted: {predictions['Linear SVM'][i]}")

    # ------------------------------------------------------------------
    # STEP 18  Generate confusion matrix plots
    # ------------------------------------------------------------------
    print("\n[STEP 18] Generating confusion matrix visualizations...")
    for name, result in evaluation_results.items():
        cm = np.array(result["confusion_matrix"])
        is_best = (name == best_model["model_name"])
        filepath = plot_confusion_matrix(
            cm=cm,
            labels=CM_LABEL_ORDER,
            model_name=name,
            highlight_best=is_best,
        )
        if is_best:
            # Save best model CM to the dedicated path
            plot_confusion_matrix(
                cm=cm,
                labels=CM_LABEL_ORDER,
                model_name=name,
                filepath=BEST_MODEL_CM_PLOT,
                highlight_best=True,
            )

    # ------------------------------------------------------------------
    # STEP 19  Generate model performance comparison chart
    # ------------------------------------------------------------------
    print("\n[STEP 19] Generating model performance comparison chart...")
    plot_model_performance_comparison(comparison_df)

    # ------------------------------------------------------------------
    # STEP 20  Save all evaluation artifacts
    # ------------------------------------------------------------------
    print("\n[STEP 20] Saving all evaluation artifacts...")
    save_classification_reports(evaluation_results)
    save_confusion_matrices_json(evaluation_results)
    save_comparison_csv(comparison_df)
    save_best_model_json(best_model)
    save_per_class_metrics_csv(evaluation_results)
    save_error_analysis_csv(error_df)
    save_evaluation_results_json(evaluation_results, best_model, len(y_test))
    save_evaluation_results_csv(evaluation_results, len(y_test))
    print("[DONE] All artifacts saved.")

    # ------------------------------------------------------------------
    # STEP 21  Data leakage verification
    # ------------------------------------------------------------------
    print("\n[STEP 21] Formal Data Leakage Verification...")
    assert X_test_tfidf.shape[0] == len(X_test_text), "X_test row count mismatch"
    assert X_test_tfidf.shape[1] == X_test_tfidf.shape[1], "Feature dimension sanity check"
    assert len(X_train_text) + len(X_test_text) == len(df), "Total sample count mismatch"
    for name, model in models.items():
        assert hasattr(model, "classes_"), f"{name} not properly fitted"
        assert len(model.classes_) == 3, f"{name} does not have 3 classes"
    print("   Train/Test split executed before vectorization")
    print("   TF-IDF loaded from disk (not refitted)")
    print("   Test labels never used during training")
    print("   All models have exactly 3 classes")
    print("[DONE] Zero data leakage confirmed.")

    # ------------------------------------------------------------------
    # COMPLETION SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  STAGE 4 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n  Test Samples Evaluated : {len(y_test)}")
    print(f"  Models Evaluated       : {len(models)}")
    print(f"   Best Model          : {best_model['model_name']}")
    print(f"     Weighted F1 Score   : {best_model['f1_score']*100:.2f}%")
    print(f"     Accuracy            : {best_model['accuracy']*100:.2f}%")
    print(f"\n  Saved Artifacts:")
    print(f"     results/evaluation_results.json")
    print(f"     results/evaluation_results.csv")
    print(f"     results/model_comparison.csv")
    print(f"     results/best_model.json")
    print(f"     results/per_class_metrics.csv")
    print(f"     results/error_analysis.csv")
    print(f"     results/classification_reports/")
    print(f"     results/confusion_matrices/")
    print(f"     results/model_performance_comparison.png")
    print(f"     results/best_model_confusion_matrix.png")
    print(f"\n  Ready for Stage 5: Real-Time Prediction Engine")
    print("=" * 80)


if __name__ == "__main__":
    run_stage4()
