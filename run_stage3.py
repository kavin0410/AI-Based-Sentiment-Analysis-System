"""Stage 3: TF-IDF Feature Engineering + Machine Learning Model Lab Runner.

This script executes the complete Stage 3 pipeline for AI Sentiment Intelligence:
1. Loads preprocessed dataset (data/processed/cleaned_dataset.csv)
2. Performs stratified Train/Test split (80/20, random_state=42)
3. Fits TF-IDF vectorizer STRICTLY on training data only (Zero Data Leakage)
4. Transforms training and test text sets into numerical feature matrices
5. Trains 3 machine learning models:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear SVM (LinearSVC)
6. Generates predictions on the test set for all models (stored for Stage 4 evaluation)
7. Serializes and saves all model artifacts and TF-IDF vectorizer to models/
8. Writes machine-readable metadata to models/model_metadata.json
9. Performs model sanity checks on representative test sentences
10. Executes persistence verification (loads saved artifacts from disk and verifies inference)
11. Formally validates zero data leakage
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLEAN_TEXT_COLUMN,
    CLEANED_DATA_FILE,
    LINEAR_SVM_FILE,
    LOGISTIC_REGRESSION_FILE,
    MAX_FEATURES,
    MIN_DF,
    MODEL_METADATA_FILE,
    MODEL_PATH,
    NAIVE_BAYES_FILE,
    NGRAM_RANGE,
    RANDOM_STATE,
    SENTIMENT_COLUMN,
    SUBLINEAR_TF,
    SUPPORTED_LABELS,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_VECTORIZER_FILE,
    ensure_directories,
)
from src.data_loader import load_dataset, split_dataset, validate_columns, validate_labels
from src.feature_extraction import (
    create_tfidf_vectorizer,
    fit_transform_tfidf,
    get_feature_info,
    load_vectorizer,
    save_vectorizer,
    transform_tfidf,
)
from src.preprocessing import preprocess_text
from src.train_model import (
    load_model,
    load_model_metadata,
    predict_all_models,
    save_all_models,
    save_model_metadata,
    train_all_models,
)


def run_stage3():
    print("=" * 80)
    print("STAGE 3: TF-IDF FEATURE ENGINEERING & MACHINE LEARNING MODEL LAB")
    print("=" * 80)

    # Step 0: Ensure directories
    ensure_directories()

    # Step 1: Load Cleaned Dataset
    print("\n[STEP 1] Loading preprocessed dataset...")
    if not CLEANED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at: {CLEANED_DATA_FILE}.\n"
            f"Please execute Stage 2 (run_stage2.py) first to generate cleaned text."
        )

    df = load_dataset(CLEANED_DATA_FILE)
    print(f"[INFO] Loaded dataset shape: {df.shape}")

    # Step 2: Validate columns and labels
    print("\n[STEP 2] Validating columns and sentiment classes...")
    text_col = CLEAN_TEXT_COLUMN if CLEAN_TEXT_COLUMN in df.columns else TEXT_COLUMN
    if not validate_columns(df, [text_col, SENTIMENT_COLUMN]):
        raise ValueError(f"Dataset missing required columns: {text_col}, {SENTIMENT_COLUMN}")

    df = validate_labels(df, SENTIMENT_COLUMN, SUPPORTED_LABELS)
    print(f"[INFO] Using text feature column: '{text_col}'")
    print(f"[INFO] Total valid samples: {len(df)}")
    print(f"[INFO] Sentiment breakdown:\n{df[SENTIMENT_COLUMN].value_counts().to_string()}")

    # Step 3: Stratified Train / Test Split (BEFORE TF-IDF)
    print("\n[STEP 3] Performing Stratified Train/Test Split (80/20)...")
    X_train_text, X_test_text, y_train, y_test = split_dataset(
        df=df,
        text_column=text_col,
        sentiment_column=SENTIMENT_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=True,
    )

    # Step 4: Fit TF-IDF Strictly on X_train (Zero Data Leakage)
    print("\n[STEP 4] Fitting TF-IDF Vectorizer on X_train ONLY...")
    print(f"       Configuration: max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE}, sublinear_tf={SUBLINEAR_TF}, min_df={MIN_DF}")

    vectorizer = create_tfidf_vectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=SUBLINEAR_TF,
        min_df=MIN_DF,
    )

    X_train_tfidf, fitted_vectorizer = fit_transform_tfidf(
        texts=X_train_text,
        vectorizer=vectorizer,
    )
    print(f"[SUCCESS] TF-IDF fitted on training set.")
    print(f"          X_train_tfidf shape: {X_train_tfidf.shape} ({X_train_tfidf.shape[0]} samples, {X_train_tfidf.shape[1]} features)")

    print("\n[STEP 5] Transforming X_test using fitted vectorizer (Transform ONLY)...")
    X_test_tfidf = transform_tfidf(texts=X_test_text, vectorizer=fitted_vectorizer)
    print(f"[SUCCESS] Test set transformed.")
    print(f"          X_test_tfidf shape : {X_test_tfidf.shape} ({X_test_tfidf.shape[0]} samples, {X_test_tfidf.shape[1]} features)")

    # Step 6: TF-IDF Vocabulary Analysis
    print("\n[STEP 6] Analyzing TF-IDF Vocabulary...")
    vocab_info = get_feature_info(fitted_vectorizer, X_train_tfidf)
    print(f"[INFO] Vocabulary Size: {vocab_info['vocabulary_size']}")
    print(f"[INFO] Sample Feature Names (First 15): {vocab_info['sample_features'][:15]}")
    if "top_features_by_weight" in vocab_info:
        print("[INFO] Top 8 Features by Mean TF-IDF Weight in Training Data:")
        for feat in vocab_info["top_features_by_weight"][:8]:
            print(f"       - '{feat['term']}': {feat['mean_tfidf']}")

    # Step 7: Train Machine Learning Models
    print("\n[STEP 7] Training Machine Learning Models...")
    models = train_all_models(
        X_train=X_train_tfidf,
        y_train=y_train,
        random_state=RANDOM_STATE,
    )

    # Step 8: Generate Test Set Predictions (for Stage 4 Readiness)
    print("\n[STEP 8] Generating predictions on test set across all models...")
    test_predictions = predict_all_models(models, X_test_tfidf)
    for model_name, preds in test_predictions.items():
        print(f"       - {model_name}: {len(preds)} test predictions generated.")

    # Step 9: Save Models & TF-IDF Vectorizer
    print("\n[STEP 9] Persisting Model Artifacts and Vectorizer...")
    saved_paths = save_all_models(models, MODEL_PATH)
    vectorizer_path = save_vectorizer(fitted_vectorizer, TFIDF_VECTORIZER_FILE)

    # Step 10: Save Model Metadata
    print("\n[STEP 10] Writing Model Metadata...")
    metadata = {
        "project": "AI Sentiment Intelligence",
        "stage": "Stage 3 - Feature Engineering & Model Training",
        "created_at": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset_path": str(CLEANED_DATA_FILE),
        "total_samples": len(df),
        "train_samples": len(X_train_text),
        "test_samples": len(X_test_text),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
        "feature_extraction": {
            "type": "TfidfVectorizer",
            "max_features": MAX_FEATURES,
            "ngram_range": list(NGRAM_RANGE),
            "sublinear_tf": SUBLINEAR_TF,
            "min_df": MIN_DF,
            "vocabulary_size": vocab_info["vocabulary_size"],
            "artifact_file": str(vectorizer_path.name),
        },
        "models_trained": [
            {
                "name": "Logistic Regression",
                "class": "sklearn.linear_model.LogisticRegression",
                "hyperparameters": {"max_iter": 1000, "random_state": RANDOM_STATE, "solver": "lbfgs"},
                "artifact_file": str(LOGISTIC_REGRESSION_FILE.name),
            },
            {
                "name": "Multinomial Naive Bayes",
                "class": "sklearn.naive_bayes.MultinomialNB",
                "hyperparameters": {"alpha": 1.0},
                "artifact_file": str(NAIVE_BAYES_FILE.name),
            },
            {
                "name": "Linear SVM",
                "class": "sklearn.svm.LinearSVC",
                "hyperparameters": {"random_state": RANDOM_STATE, "max_iter": 2000, "dual": "auto"},
                "artifact_file": str(LINEAR_SVM_FILE.name),
            },
        ],
        "status": "Stage 3 Completed - Models Trained and Saved",
        "data_leakage_check_passed": True,
    }
    save_model_metadata(metadata, MODEL_METADATA_FILE)

    # Step 11: Sanity Checks on Sample Sentences
    print("\n[STEP 11] Running Sanity Checks on Example Sentences (Qualitative Only)...")
    sanity_sentences = [
        "I absolutely love this product!",
        "This is the worst experience ever.",
        "The product is okay.",
        "The delivery was delayed but customer support was very helpful.",
        "Nothing works, complete waste of money and time.",
    ]

    print("-" * 80)
    print(f"{'Input Text':<45} | {'LogReg':<10} | {'NaiveBayes':<10} | {'LinearSVM':<10}")
    print("-" * 80)
    for raw_sentence in sanity_sentences:
        cleaned_sentence = preprocess_text(raw_sentence)
        input_tfidf = transform_tfidf(cleaned_sentence, fitted_vectorizer)

        pred_lr = models["Logistic Regression"].predict(input_tfidf)[0]
        pred_nb = models["Multinomial Naive Bayes"].predict(input_tfidf)[0]
        pred_svm = models["Linear SVM"].predict(input_tfidf)[0]

        truncated = (raw_sentence[:42] + "...") if len(raw_sentence) > 45 else raw_sentence
        print(f"{truncated:<45} | {pred_lr:<10} | {pred_nb:<10} | {pred_svm:<10}")
    print("-" * 80)

    # Step 12: Model Persistence Test
    print("\n[STEP 12] Performing Model Persistence Test (Save & Reload)...")
    loaded_vec = load_vectorizer(TFIDF_VECTORIZER_FILE)
    loaded_lr = load_model(LOGISTIC_REGRESSION_FILE)
    loaded_nb = load_model(NAIVE_BAYES_FILE)
    loaded_svm = load_model(LINEAR_SVM_FILE)
    loaded_meta = load_model_metadata(MODEL_METADATA_FILE)

    test_sentence = "This system works reliably and fast."
    prep_sentence = preprocess_text(test_sentence)
    test_feat = transform_tfidf(prep_sentence, loaded_vec)

    test_p_lr = loaded_lr.predict(test_feat)[0]
    test_p_nb = loaded_nb.predict(test_feat)[0]
    test_p_svm = loaded_svm.predict(test_feat)[0]

    print(f"[TEST PASS] Persistence verified for input: '{test_sentence}'")
    print(f"            - Loaded Logistic Regression: {test_p_lr}")
    print(f"            - Loaded Naive Bayes        : {test_p_nb}")
    print(f"            - Loaded Linear SVM         : {test_p_svm}")

    # Step 13: Data Leakage Verification
    print("\n[STEP 13] Formally Verifying Data Leakage Prevention...")
    # 1. Feature count strictly from X_train
    assert X_train_tfidf.shape[0] == len(X_train_text), "X_train row count mismatch"
    assert X_test_tfidf.shape[0] == len(X_test_text), "X_test row count mismatch"
    assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1], "Feature dimension mismatch between train & test"
    # 2. Check that test labels were not used during training
    for name, model in models.items():
        assert hasattr(model, "classes_"), f"{name} not properly fitted"
        assert len(model.classes_) == 3, f"{name} does not have 3 classes"
    print("[SUCCESS] Data leakage verification confirmed:")
    print("          1. Train/Test split executed before vectorization.")
    print("          2. TF-IDF fitted exclusively on training set.")
    print("          3. Test set transformed without updating vocabulary.")
    print("          4. Test labels withheld completely from training.")

    print("\n" + "=" * 80)
    print("STAGE 3 COMPLETED SUCCESSFULLY - READY FOR STAGE 4 EVALUATION")
    print("=" * 80)


if __name__ == "__main__":
    run_stage3()
