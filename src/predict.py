"""Prediction Module for AI Sentiment Intelligence (Stage 5).

Provides real-time sentiment prediction using the dynamically selected best model
and pre-fitted TF-IDF vectorizer.

Pipeline Flow:
    Raw Text -> Input Validation -> Stage 2 NLP Preprocessing -> Saved TF-IDF Transform
    -> Best Model Predict -> Score/Probability Extraction -> Structured Output Object

Handles:
- Logistic Regression / Naive Bayes: predict_proba() -> probability score (0.0 to 1.0)
- Linear SVM: decision_function() -> decision_score (labeled strictly as decision_score)
- In-memory session history management with configurable limit (default 50)
- CSV history export
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

# Import configuration with fallback
try:
    from config import (
        MAX_INPUT_LENGTH,
        PREDICTION_HISTORY_LIMIT,
        SUPPORTED_LABELS,
    )
except ImportError:
    try:
        from src.config import (
            MAX_INPUT_LENGTH,
            PREDICTION_HISTORY_LIMIT,
            SUPPORTED_LABELS,
        )
    except ImportError:
        MAX_INPUT_LENGTH = 5000
        PREDICTION_HISTORY_LIMIT = 50
        SUPPORTED_LABELS = ["Positive", "Negative", "Neutral"]

# Import module functions
try:
    from src.feature_extraction import transform_tfidf
    from src.model_loader import load_best_model, load_vectorizer
    from src.preprocessing import preprocess_text
except ImportError:
    from feature_extraction import transform_tfidf
    from model_loader import load_best_model, load_vectorizer
    from preprocessing import preprocess_text


def validate_input_text(
    text: Any, max_length: int = MAX_INPUT_LENGTH
) -> Tuple[bool, Optional[str]]:
    """Validate user input text against empty/whitespace and maximum length rules.

    Args:
        text: Raw input text from user.
        max_length: Maximum allowed character length.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
    """
    if text is None:
        return False, "Please enter some text to analyze."

    if not isinstance(text, str):
        text = str(text)

    stripped = text.strip()
    if not stripped:
        return False, "Please enter some text to analyze."

    if len(text) > max_length:
        return (
            False,
            f"Input exceeds maximum allowed length of {max_length} characters "
            f"(current length: {len(text)} characters). Please shorten your input.",
        )

    return True, None


def predict_sentiment(
    text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end real-time sentiment prediction on a single text string.

    Args:
        text: User input text string.
        model: Loaded model instance. If None, dynamically loads the best model.
        vectorizer: Loaded TF-IDF vectorizer. If None, loads saved vectorizer.
        model_name: Optional string label for the model name.

    Returns:
        Dictionary containing prediction result, scores, preprocessed text, and metadata.
    """
    # 1. Input Validation
    is_valid, error_msg = validate_input_text(text)
    if not is_valid:
        return {
            "valid": False,
            "error": error_msg,
            "text": text if text is not None else "",
            "timestamp": datetime.now().isoformat(),
        }

    # 2. Dynamic Artifact Loading if not provided
    if model is None or model_name is None:
        model, loaded_name, _ = load_best_model()
        model_name = model_name or loaded_name

    if vectorizer is None:
        vectorizer = load_vectorizer()

    # 3. Stage 2 NLP Preprocessing
    clean_text = preprocess_text(text)
    # Fallback if text becomes empty after stripping stop words (e.g. input was just "the a an")
    feature_text = clean_text if clean_text.strip() else text.lower().strip()

    # 4. TF-IDF Transformation (Transform ONLY - zero data leakage)
    X_tfidf = transform_tfidf(feature_text, vectorizer)

    # 5. Model Prediction
    prediction = model.predict(X_tfidf)[0]

    # 6. Score & Confidence Handling
    scores_dict: Dict[str, float] = {}
    score_val: float = 0.0
    score_type: str = "probability"

    if hasattr(model, "predict_proba"):
        # Logistic Regression or Multinomial Naive Bayes
        proba_arr = model.predict_proba(X_tfidf)[0]
        classes = getattr(model, "classes_", SUPPORTED_LABELS)
        scores_dict = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, proba_arr)}

        # Score for predicted class
        score_val = scores_dict.get(str(prediction), round(float(max(proba_arr)), 4))
        score_type = "probability"
    elif hasattr(model, "decision_function"):
        # Linear SVM (LinearSVC)
        dec_arr = model.decision_function(X_tfidf)[0]
        classes = getattr(model, "classes_", SUPPORTED_LABELS)
        if hasattr(dec_arr, "__len__") and len(dec_arr) == len(classes):
            scores_dict = {str(cls): round(float(sc), 4) for cls, sc in zip(classes, dec_arr)}
            score_val = scores_dict.get(str(prediction), 0.0)
        else:
            score_val = round(float(dec_arr), 4)
            scores_dict = {str(prediction): score_val}
        score_type = "decision_score"
    else:
        score_val = 1.0
        scores_dict = {str(prediction): 1.0}
        score_type = "decision_score"

    vocab_size = len(vectorizer.get_feature_names_out()) if hasattr(vectorizer, "get_feature_names_out") else 0

    return {
        "valid": True,
        "text": text,
        "processed_text": clean_text,
        "sentiment": str(prediction),
        "model_name": model_name,
        "score": score_val,
        "score_type": score_type,
        "probabilities": scores_dict,
        "timestamp": datetime.now().isoformat(),
        "vocab_size": vocab_size,
    }


def predict_batch(
    texts: List[str],
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute sentiment prediction for a batch of text strings.

    Args:
        texts: List of raw input text strings.
        model: Optional model instance.
        vectorizer: Optional vectorizer instance.
        model_name: Optional model name label.

    Returns:
        List of prediction result dictionaries.
    """
    if model is None or model_name is None:
        model, model_name, _ = load_best_model()
    if vectorizer is None:
        vectorizer = load_vectorizer()

    results = []
    for txt in texts:
        res = predict_sentiment(txt, model=model, vectorizer=vectorizer, model_name=model_name)
        results.append(res)
    return results


def format_prediction_result(result: Dict[str, Any]) -> str:
    """Format prediction result dictionary into a clean text string for display.

    Args:
        result: Prediction result dictionary from predict_sentiment.

    Returns:
        Formatted multi-line display string.
    """
    if not result.get("valid", False):
        return f"[ERROR] {result.get('error', 'Invalid input')}"

    sentiment = result.get("sentiment", "Unknown")
    model_name = result.get("model_name", "Model")
    score = result.get("score", 0.0)
    score_type = result.get("score_type", "score")

    if score_type == "probability":
        score_str = f"Confidence: {score * 100:.1f}%"
    else:
        score_str = f"Decision Score: {score:.4f}"

    lines = [
        f"Sentiment   : {sentiment}",
        f"Model       : {model_name}",
        f"{score_str}",
        f"Clean Text  : '{result.get('processed_text', '')}'",
        f"Timestamp   : {result.get('timestamp', '')}",
    ]
    return "\n".join(lines)


class PredictionHistoryManager:
    """In-memory session prediction history manager with configurable size limit."""

    def __init__(self, max_limit: int = PREDICTION_HISTORY_LIMIT):
        self.max_limit = max_limit
        self.history: List[Dict[str, Any]] = []

    def add_prediction(self, result: Dict[str, Any]) -> None:
        """Add a valid prediction result to history, enforcing size limit."""
        if not result.get("valid", False):
            return

        entry = {
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "text": result.get("text", ""),
            "processed_text": result.get("processed_text", ""),
            "sentiment": result.get("sentiment", ""),
            "model": result.get("model_name", ""),
            "score": result.get("score", 0.0),
            "score_type": result.get("score_type", "probability"),
        }

        self.history.append(entry)

        # Enforce history size limit
        if len(self.history) > self.max_limit:
            self.history = self.history[-self.max_limit:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieve full prediction history list."""
        return list(self.history)

    def clear(self) -> None:
        """Clear all session prediction history."""
        self.history.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert prediction history to pandas DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=[
                "timestamp", "text", "processed_text", "sentiment", "model", "score", "score_type"
            ])
        return pd.DataFrame(self.history)

    def to_csv(self) -> str:
        """Export prediction history as CSV string."""
        df = self.to_dataframe()
        return df.to_csv(index=False)
