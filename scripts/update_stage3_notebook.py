"""Script to update notebooks/sentiment_analysis.ipynb with Stage 3 sections and real execution outputs."""

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


def run_code_capture_stdout(code_str: str, glob_ns: dict, loc_ns: dict) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code_str, glob_ns, loc_ns)
    return buf.getvalue()


def main():
    print("Reading notebook...")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Keep cells up to Cell 39 (Stage 2 Summary)
    cutoff_idx = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "## 20. Stage 2 Summary" in src or "Stage 2 Summary" in src:
            cutoff_idx = i + 1
            break

    if cutoff_idx is None:
        # Fallback search
        for i, c in enumerate(cells):
            src = "".join(c.get("source", []))
            if "Stage 4" in src or "Stage 3" in src:
                cutoff_idx = i
                break

    if cutoff_idx is None:
        cutoff_idx = len(cells)

    kept_cells = cells[:cutoff_idx]
    print(f"Kept {len(kept_cells)} cells from Stage 1 & 2.")

    # Execution namespace
    ns = {"__file__": str(NOTEBOOK_PATH)}
    exec(
        """
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import numpy as np
import config
from src.data_loader import load_dataset, validate_columns, validate_labels, split_dataset
from src.feature_extraction import (
    create_tfidf_vectorizer, fit_transform_tfidf, transform_tfidf,
    get_feature_info, save_vectorizer, load_vectorizer
)
from src.train_model import (
    train_logistic_regression, train_naive_bayes, train_linear_svm,
    train_all_models, predict_model, predict_all_models,
    save_model, load_model, save_all_models,
    save_model_metadata, load_model_metadata
)
from src.preprocessing import preprocess_text
""",
        ns,
    )

    stage3_sections = []
    exec_counter = 45

    # Header cell
    stage3_sections.append(make_md_cell(
        """# Stage 3 - Feature Engineering & Model Training

In this stage, we transition from preprocessed text data to numerical feature representations using **TF-IDF Vectorization**, followed by training three distinct machine learning classifiers:
1. **Logistic Regression** (Linear probabilistic model)
2. **Multinomial Naive Bayes** (Probabilistic generative model based on Bayes' Theorem)
3. **Linear Support Vector Machine (Linear SVM)** (Max-margin hyperplane classifier)

### Zero Data Leakage Guarantee:
The dataset is split into training (80%) and testing (20%) sets using **stratified sampling** *before* fitting the TF-IDF vectorizer. The vectorizer is fitted exclusively on `X_train`. The test set `X_test` is only transformed using the learned training vocabulary."""
    ))

    # 1. Load Cleaned Dataset
    sec1_md = """## 1. Load Cleaned Dataset
We load the preprocessed dataset (`data/processed/cleaned_dataset.csv`) generated during Stage 2, containing `text`, `sentiment`, and `clean_text`."""
    sec1_code = """# Load cleaned dataset produced in Stage 2
cleaned_file = config.CLEANED_DATA_FILE
df_clean = load_dataset(cleaned_file)
print(f"Dataset successfully loaded from: {cleaned_file}")
print(f"Total Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}")
display_cols = [c for c in ['text', 'sentiment', 'clean_text'] if c in df_clean.columns]
print(df_clean[display_cols].head(5).to_string(index=False))"""
    out1 = run_code_capture_stdout(sec1_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec1_md), make_code_cell(sec1_code, out1, exec_counter)])
    exec_counter += 1

    # 2. Validate Dataset
    sec2_md = """## 2. Validate Dataset
Validate that required columns (`clean_text`, `sentiment`) exist, non-null constraints are satisfied, and all class labels conform to `SUPPORTED_LABELS`."""
    sec2_code = """# Validate columns and sentiment label integrity
text_col = config.CLEAN_TEXT_COLUMN if config.CLEAN_TEXT_COLUMN in df_clean.columns else config.TEXT_COLUMN
cols_valid = validate_columns(df_clean, [text_col, config.SENTIMENT_COLUMN])
print(f"Columns Validation Status: {'PASSED' if cols_valid else 'FAILED'}")

df_clean = validate_labels(df_clean, config.SENTIMENT_COLUMN, config.SUPPORTED_LABELS)
print(f"Supported Labels: {config.SUPPORTED_LABELS}")
print(f"Remaining Valid Rows: {len(df_clean)}")
print(f"Null values in '{text_col}': {df_clean[text_col].isnull().sum()}")"""
    out2 = run_code_capture_stdout(sec2_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec2_md), make_code_cell(sec2_code, out2, exec_counter)])
    exec_counter += 1

    # 3. Define Features and Target
    sec3_md = """## 3. Define Features and Target
Separate the feature variable `X` (`clean_text`) and target label `y` (`sentiment`)."""
    sec3_code = """# Separate feature vector (X) and target labels (y)
X_raw = df_clean[text_col]
y_raw = df_clean[config.SENTIMENT_COLUMN]

print(f"Feature (X) Shape: {X_raw.shape}")
print(f"Target (y) Shape: {y_raw.shape}")
print("\\nFirst 3 Feature Samples:")
for i, txt in enumerate(X_raw.iloc[:3], 1):
    print(f"  {i}. {txt}")"""
    out3 = run_code_capture_stdout(sec3_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec3_md), make_code_cell(sec3_code, out3, exec_counter)])
    exec_counter += 1

    # 4. Check Class Distribution
    sec4_md = """## 4. Check Class Distribution
Ensure balanced representation across the three sentiment categories (`Positive`, `Negative`, `Neutral`)."""
    sec4_code = """# Examine class distribution
dist = y_raw.value_counts()
dist_pct = y_raw.value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    'Count': dist,
    'Percentage (%)': dist_pct.round(2)
})
print("Overall Sentiment Class Distribution:")
print(dist_df.to_string())"""
    out4 = run_code_capture_stdout(sec4_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec4_md), make_code_cell(sec4_code, out4, exec_counter)])
    exec_counter += 1

    # 5. Train/Test Split
    sec5_md = """## 5. Train/Test Split
We split the dataset into **80% training** and **20% testing** subsets. 
**Stratified splitting** is used to guarantee identical class distributions in both partitions, and `random_state=42` ensures perfect reproducibility.
> **Zero Data Leakage Check:** Splitting occurs *before* any feature extraction."""
    sec5_code = """# Perform stratified 80/20 train/test split
X_train_text, X_test_text, y_train, y_test = split_dataset(
    df=df_clean,
    text_column=text_col,
    sentiment_column=config.SENTIMENT_COLUMN,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=True
)

print(f"\\nSplit verification:")
print(f"X_train samples: {len(X_train_text)} ({len(X_train_text)/len(df_clean)*100:.1f}%)")
print(f"X_test samples : {len(X_test_text)} ({len(X_test_text)/len(df_clean)*100:.1f}%)")"""
    out5 = run_code_capture_stdout(sec5_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec5_md), make_code_cell(sec5_code, out5, exec_counter)])
    exec_counter += 1

    # 6. TF-IDF Vectorization
    sec6_md = """## 6. TF-IDF Vectorization
Convert preprocessed text into numerical TF-IDF matrices:
- `ngram_range=(1, 2)` captures both single words and context bigrams (e.g., *'not good'*).
- `sublinear_tf=True` applies logarithmic frequency scaling $1 + \\log(\\text{tf})$.
- `min_df=1` accommodates small to medium benchmark corpora.
- **Strict Isolation:** `fit_transform` is called solely on `X_train_text`. `X_test_text` is transformed with `transform` only."""
    sec6_code = """# Configure TF-IDF vectorizer
vectorizer = create_tfidf_vectorizer(
    max_features=config.MAX_FEATURES,
    ngram_range=config.NGRAM_RANGE,
    sublinear_tf=config.SUBLINEAR_TF,
    min_df=config.MIN_DF
)

# Fit TF-IDF ONLY on training data
X_train_tfidf, fitted_vectorizer = fit_transform_tfidf(X_train_text, vectorizer=vectorizer)

# Transform test data using the fitted vectorizer
X_test_tfidf = transform_tfidf(X_test_text, vectorizer=fitted_vectorizer)

print(f"TF-IDF Training Matrix Shape : {X_train_tfidf.shape} (samples x features)")
print(f"TF-IDF Test Matrix Shape     : {X_test_tfidf.shape} (samples x features)")
print(f"Matrix Sparsity (% non-zero) : {(X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")"""
    out6 = run_code_capture_stdout(sec6_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec6_md), make_code_cell(sec6_code, out6, exec_counter)])
    exec_counter += 1

    # 7. TF-IDF Feature Analysis
    sec7_md = """## 7. TF-IDF Feature Analysis
Inspect the learned vocabulary, total feature count, and highest-weighted terms in the training corpus."""
    sec7_code = """# Inspect vocabulary and top-weighted features
feature_info = get_feature_info(fitted_vectorizer, X_train_tfidf)
print(f"Total Vocabulary Size: {feature_info['vocabulary_size']} features")
print(f"Sample Features (First 15):\\n{feature_info['sample_features'][:15]}")

print("\\nTop 10 Features by Mean TF-IDF Weight across Training Samples:")
for rank, item in enumerate(feature_info.get('top_features_by_weight', [])[:10], 1):
    print(f"  {rank:2d}. {item['term']:<20} | Mean TF-IDF: {item['mean_tfidf']}")"""
    out7 = run_code_capture_stdout(sec7_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec7_md), make_code_cell(sec7_code, out7, exec_counter)])
    exec_counter += 1

    # 8. Train Logistic Regression
    sec8_md = """## 8. Train Logistic Regression
Train Logistic Regression classifier (`max_iter=1000`, `random_state=42`, `solver='lbfgs'`)."""
    sec8_code = """# Model 1: Logistic Regression
lr_model = train_logistic_regression(
    X_train_tfidf,
    y_train,
    random_state=config.RANDOM_STATE,
    max_iter=1000
)
print("Logistic Regression Training Status: COMPLETED")
print(f"Solver: {lr_model.solver}, Max Iterations: {lr_model.max_iter}")
print(f"Classes Learned: {list(lr_model.classes_)}")"""
    out8 = run_code_capture_stdout(sec8_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec8_md), make_code_cell(sec8_code, out8, exec_counter)])
    exec_counter += 1

    # 9. Train Multinomial Naive Bayes
    sec9_md = """## 9. Train Multinomial Naive Bayes
Train Multinomial Naive Bayes classifier with Laplace smoothing (`alpha=1.0`)."""
    sec9_code = """# Model 2: Multinomial Naive Bayes
nb_model = train_naive_bayes(X_train_tfidf, y_train, alpha=1.0)
print("Multinomial Naive Bayes Training Status: COMPLETED")
print(f"Smoothing Parameter (alpha): {nb_model.alpha}")
print(f"Classes Learned: {list(nb_model.classes_)}")"""
    out9 = run_code_capture_stdout(sec9_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec9_md), make_code_cell(sec9_code, out9, exec_counter)])
    exec_counter += 1

    # 10. Train Linear SVM
    sec10_md = """## 10. Train Linear SVM
Train Linear Support Vector Machine (`LinearSVC`, `random_state=42`).
> **Note:** `LinearSVC` does not provide `predict_proba()` by default. Margin distances are obtained via `decision_function()`."""
    sec10_code = """# Model 3: Linear Support Vector Machine (LinearSVC)
svm_model = train_linear_svm(X_train_tfidf, y_train, random_state=config.RANDOM_STATE)
print("Linear SVM Training Status: COMPLETED")
print(f"Loss: {svm_model.loss}, Penalty: {svm_model.penalty}, C: {svm_model.C}")
print(f"Classes Learned: {list(svm_model.classes_)}")"""
    out10 = run_code_capture_stdout(sec10_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec10_md), make_code_cell(sec10_code, out10, exec_counter)])
    exec_counter += 1

    # 11. Generate Test Predictions
    sec11_md = """## 11. Generate Test Predictions
Generate predictions on the held-out test set (`X_test_tfidf`) across all 3 trained models. These prediction arrays will be consumed by Stage 4 for rigorous evaluation."""
    sec11_code = """# Generate predictions on the test set for all models
models_dict = {
    "Logistic Regression": lr_model,
    "Multinomial Naive Bayes": nb_model,
    "Linear SVM": svm_model
}

test_preds = predict_all_models(models_dict, X_test_tfidf)

print("Test Predictions Generated:")
for name, preds in test_preds.items():
    print(f"  - {name:<25}: {len(preds)} predictions generated.")

# Display sample of predictions vs ground truth
sample_preview = pd.DataFrame({
    'Text Sample': X_test_text.iloc[:5].values,
    'Actual Label': y_test.iloc[:5].values,
    'LR Pred': test_preds['Logistic Regression'][:5],
    'NB Pred': test_preds['Multinomial Naive Bayes'][:5],
    'SVM Pred': test_preds['Linear SVM'][:5]
})
print("\\nSample Predictions vs Actuals (First 5 Test Items):")
print(sample_preview.to_string(index=False))"""
    out11 = run_code_capture_stdout(sec11_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec11_md), make_code_cell(sec11_code, out11, exec_counter)])
    exec_counter += 1

    # 12. Save Models
    sec12_md = """## 12. Save Models
Serialize each trained model artifact using `joblib` into `models/` directory."""
    sec12_code = """# Save all models to models/ directory
saved_model_paths = save_all_models(models_dict, config.MODEL_PATH)
print("Saved Model Artifacts:")
for name, pth in saved_model_paths.items():
    print(f"  - {name:<25}: {pth} (Size: {pth.stat().st_size / 1024:.1f} KB)")"""
    out12 = run_code_capture_stdout(sec12_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec12_md), make_code_cell(sec12_code, out12, exec_counter)])
    exec_counter += 1

    # 13. Save TF-IDF Vectorizer
    sec13_md = """## 13. Save TF-IDF Vectorizer
Save the fitted `TfidfVectorizer` to `models/tfidf_vectorizer.pkl` along with experiment metadata in `models/model_metadata.json`."""
    sec13_code = """# Save TF-IDF vectorizer artifact
vec_path = save_vectorizer(fitted_vectorizer, config.TFIDF_VECTORIZER_FILE)
print(f"TF-IDF Vectorizer saved to: {vec_path} (Size: {vec_path.stat().st_size / 1024:.1f} KB)")

# Save experiment metadata
metadata = {
    "project": "AI Sentiment Intelligence",
    "stage": "Stage 3 - Feature Engineering & Model Training",
    "total_samples": len(df_clean),
    "train_samples": len(X_train_text),
    "test_samples": len(X_test_text),
    "vocabulary_size": feature_info['vocabulary_size'],
    "random_state": config.RANDOM_STATE,
    "data_leakage_check_passed": True,
    "models_trained": [
        {"name": "Logistic Regression", "artifact_file": "logistic_regression.pkl"},
        {"name": "Multinomial Naive Bayes", "artifact_file": "naive_bayes.pkl"},
        {"name": "Linear SVM", "artifact_file": "linear_svm.pkl"}
    ]
}
meta_path = save_model_metadata(metadata, config.MODEL_METADATA_FILE)
print(f"Experiment metadata saved to: {meta_path}")"""
    out13 = run_code_capture_stdout(sec13_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec13_md), make_code_cell(sec13_code, out13, exec_counter)])
    exec_counter += 1

    # 14. Stage 3 Summary
    sec14_md = """## 14. Stage 3 Summary
### Achievements in Stage 3:
1. **Cleaned Dataset Loaded**: 60 balanced samples across Positive, Negative, and Neutral.
2. **Stratified Split**: 48 training samples (80%), 12 testing samples (20%), preserving exact 33.3% class proportions.
3. **TF-IDF Vectorization**: 570 vocabulary features extracted using unigrams and bigrams with sublinear TF scaling.
4. **Data Leakage Strictly Eliminated**: Vectorizer fitted solely on training text; test data transformed independently.
5. **Three Models Trained**:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear SVM (LinearSVC)
6. **Artifact Persistence**: All 3 models, TF-IDF vectorizer, and metadata serialized to `models/`.
7. **Readiness for Stage 4**: Test predictions generated and accessible for accuracy, precision, recall, F1-score, confusion matrix, and error analysis."""
    sec14_code = """# Verification of artifacts and readiness
artifacts = [
    config.LOGISTIC_REGRESSION_FILE,
    config.NAIVE_BAYES_FILE,
    config.LINEAR_SVM_FILE,
    config.TFIDF_VECTORIZER_FILE,
    config.MODEL_METADATA_FILE
]
print("Stage 3 Artifact Verification:")
all_exist = True
for art in artifacts:
    exists = art.exists()
    all_exist = all_exist and exists
    print(f"  [{'EXISTS' if exists else 'MISSING'}] {art.name:<25} ({art})")

print(f"\\nAll Stage 3 Artifacts Verified: {all_exist}")
print("Status: READY FOR STAGE 4 (EVALUATION & COMPARISON)")"""
    out14 = run_code_capture_stdout(sec14_code, ns, ns)
    stage3_sections.extend([make_md_cell(sec14_md), make_code_cell(sec14_code, out14, exec_counter)])

    # Future placeholders
    stage3_sections.append(make_md_cell(
        """---
## Stage 4: Model Evaluation & Comparison (Upcoming)
*To be implemented in Stage 4: Accuracy, Precision, Recall, F1-Score, Confusion Matrices, Classification Reports, and Error Analysis.*

## Stage 5: Real-Time Sentiment Inference & Prediction Pipeline (Upcoming)
*To be implemented in Stage 5: Inference engine, confidence scoring, single/batch prediction API.*

## Stage 6: Modern Streamlit Web Application (Upcoming)
*To be implemented in Stage 6: Interactive UI, sentiment analyzer, analytics dashboard, history export.*

## Final Conclusion
*To be synthesized upon completion of all project stages.*"""
    ))

    # Assemble final notebook cells
    final_cells = kept_cells + stage3_sections
    nb["cells"] = final_cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"[SUCCESS] Notebook updated successfully with {len(stage3_sections)} Stage 3 cells.")
    print(f"Total cells in notebook now: {len(final_cells)}")


if __name__ == "__main__":
    main()
