"""Model Training Module for AI Sentiment Intelligence.

Handles training, prediction, serialization, and metadata management for
three machine learning models:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)

Viva Note:
    - Logistic Regression: A linear model using the logistic/softmax function.
      Well-suited for text classification with high-dimensional TF-IDF sparse features,
      providing calibrated class probability estimates.
    - Multinomial Naive Bayes: A probabilistic classifier based on Bayes' Theorem with
      the assumption of feature independence. Computationally fast and effective on word counts/TF-IDF.
    - Linear SVM: Finds the maximum-margin hyperplane separating classes. Extremely
      effective in high-dimensional text feature spaces. Note: LinearSVC does NOT provide
      predict_proba() by default; decision_function() represents confidence margins.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Import configuration constants with fallback
try:
    from config import (
        LINEAR_SVM_FILE,
        LOGISTIC_REGRESSION_FILE,
        MODEL_METADATA_FILE,
        MODEL_PATH,
        NAIVE_BAYES_FILE,
        RANDOM_STATE,
        SUPPORTED_LABELS,
    )
except ImportError:
    try:
        from src.config import (
            LINEAR_SVM_FILE,
            LOGISTIC_REGRESSION_FILE,
            MODEL_METADATA_FILE,
            MODEL_PATH,
            NAIVE_BAYES_FILE,
            RANDOM_STATE,
            SUPPORTED_LABELS,
        )
    except ImportError:
        RANDOM_STATE = 42
        MODEL_PATH = Path("models")
        LOGISTIC_REGRESSION_FILE = MODEL_PATH / "logistic_regression.pkl"
        NAIVE_BAYES_FILE = MODEL_PATH / "naive_bayes.pkl"
        LINEAR_SVM_FILE = MODEL_PATH / "linear_svm.pkl"
        MODEL_METADATA_FILE = MODEL_PATH / "model_metadata.json"
        SUPPORTED_LABELS = ["Positive", "Negative", "Neutral"]


def train_logistic_regression(
    X_train: Any,
    y_train: Any,
    random_state: int = RANDOM_STATE,
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogisticRegression:
    """Train and return a Logistic Regression classifier for sentiment analysis.

    Args:
        X_train: Training feature matrix (TF-IDF features).
        y_train: Training sentiment labels.
        random_state: Random seed for solver reproducibility. Defaults to 42.
        max_iter: Maximum number of iterations for the solver to converge.
        C: Inverse of regularization strength (smaller values specify stronger regularization).

    Returns:
        Fitted LogisticRegression model instance.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(
    X_train: Any,
    y_train: Any,
    alpha: float = 1.0,
) -> MultinomialNB:
    """Train and return a Multinomial Naive Bayes classifier for sentiment analysis.

    Args:
        X_train: Training feature matrix (TF-IDF features).
        y_train: Training sentiment labels.
        alpha: Additive (Laplace/Lidstone) smoothing parameter (1.0 = Laplace smoothing).

    Returns:
        Fitted MultinomialNB model instance.
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_linear_svm(
    X_train: Any,
    y_train: Any,
    random_state: int = RANDOM_STATE,
    max_iter: int = 2000,
    C: float = 1.0,
) -> LinearSVC:
    """Train and return a Linear Support Vector Machine (LinearSVC) classifier.

    Viva Note:
        LinearSVC uses the liblinear library and is optimized for large-scale text classification.
        By default, it does not produce class probabilities via predict_proba(); instead,
        confidence scores can be obtained via decision_function().

    Args:
        X_train: Training feature matrix (TF-IDF features).
        y_train: Training sentiment labels.
        random_state: Random seed for reproducibility. Defaults to 42.
        max_iter: Maximum number of iterations for optimization.
        C: Regularization parameter.

    Returns:
        Fitted LinearSVC model instance.
    """
    model = LinearSVC(
        random_state=random_state,
        max_iter=max_iter,
        C=C,
        dual="auto",
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: Any,
    y_train: Any,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Train all three core machine learning models in a registry dictionary.

    Args:
        X_train: Training feature matrix (TF-IDF features).
        y_train: Training labels/targets.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping model names to fitted model instances:
        {
            "Logistic Regression": lr_model,
            "Multinomial Naive Bayes": nb_model,
            "Linear SVM": svm_model
        }
    """
    print("[INFO] Starting training pipeline for all 3 sentiment models...")

    models: Dict[str, Any] = {}

    # 1. Logistic Regression
    print("  -> Training Logistic Regression (max_iter=1000)...")
    models["Logistic Regression"] = train_logistic_regression(
        X_train, y_train, random_state=random_state
    )
    print("     [DONE] Logistic Regression trained successfully.")

    # 2. Multinomial Naive Bayes
    print("  -> Training Multinomial Naive Bayes (alpha=1.0)...")
    models["Multinomial Naive Bayes"] = train_naive_bayes(X_train, y_train)
    print("     [DONE] Multinomial Naive Bayes trained successfully.")

    # 3. Linear SVM
    print("  -> Training Linear SVM (LinearSVC)...")
    models["Linear SVM"] = train_linear_svm(
        X_train, y_train, random_state=random_state
    )
    print("     [DONE] Linear SVM trained successfully.")

    print("[SUCCESS] All 3 models trained successfully.")
    return models


def predict_model(model: Any, X: Any) -> np.ndarray:
    """Generate class label predictions using a trained model.

    Args:
        model: Trained scikit-learn classifier.
        X: Feature matrix (e.g. TF-IDF sparse matrix).

    Returns:
        1D numpy array of predicted sentiment class labels.
    """
    return model.predict(X)


def predict_all_models(
    models: Dict[str, Any], X: Any
) -> Dict[str, np.ndarray]:
    """Generate predictions for a dataset across all trained models.

    Args:
        models: Dictionary of trained models.
        X: Feature matrix.

    Returns:
        Dictionary mapping model name to array of predictions.
    """
    predictions: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        predictions[name] = predict_model(model, X)
    return predictions


def save_model(model: Any, filepath: Union[Path, str]) -> Path:
    """Save a trained machine learning model to disk using joblib.

    Args:
        model: Trained scikit-learn model object.
        filepath: Destination file path.

    Returns:
        Path to the saved model file.
    """
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)
    print(f"[SUCCESS] Model successfully saved to: {target_path}")
    return target_path


def load_model(filepath: Union[Path, str]) -> Any:
    """Load a saved machine learning model from disk.

    Args:
        filepath: Path to the serialized model file.

    Returns:
        Loaded scikit-learn model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {target_path.resolve()}.\n"
            f"Please run the model training stage first to generate and save the model."
        )

    model = joblib.load(target_path)
    print(f"[SUCCESS] Model loaded successfully from: {target_path}")
    return model


def save_all_models(
    models: Dict[str, Any], model_dir: Optional[Union[Path, str]] = None
) -> Dict[str, Path]:
    """Save all trained models in the registry to disk using standard naming conventions.

    Args:
        models: Dictionary of trained models {"Logistic Regression": lr, ...}.
        model_dir: Target directory. Defaults to MODEL_PATH (models/).

    Returns:
        Dictionary mapping model name to its saved file path.
    """
    if model_dir is None:
        model_dir = MODEL_PATH
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    name_to_path = {
        "Logistic Regression": model_dir / "logistic_regression.pkl",
        "Multinomial Naive Bayes": model_dir / "naive_bayes.pkl",
        "Linear SVM": model_dir / "linear_svm.pkl",
    }

    saved_paths: Dict[str, Path] = {}
    for name, model in models.items():
        dest = name_to_path.get(name, model_dir / f"{name.lower().replace(' ', '_')}.pkl")
        save_model(model, dest)
        saved_paths[name] = dest

    return saved_paths


def save_model_metadata(
    metadata: Dict[str, Any], filepath: Optional[Union[Path, str]] = None
) -> Path:
    """Save model experiment metadata to a JSON file.

    Args:
        metadata: Dictionary containing training parameters, shapes, and date.
        filepath: Destination file path. Defaults to MODEL_METADATA_FILE.

    Returns:
        Path to the saved JSON metadata file.
    """
    if filepath is None:
        filepath = MODEL_METADATA_FILE

    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Model metadata saved to: {target_path}")
    return target_path


def load_model_metadata(
    filepath: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Load model experiment metadata from a JSON file.

    Args:
        filepath: Path to the metadata file. Defaults to MODEL_METADATA_FILE.

    Returns:
        Dictionary containing metadata.

    Raises:
        FileNotFoundError: If metadata file does not exist.
    """
    if filepath is None:
        filepath = MODEL_METADATA_FILE

    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(f"Model metadata file not found at: {target_path.resolve()}")

    with open(target_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata

