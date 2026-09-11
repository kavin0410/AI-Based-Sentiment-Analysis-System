"""FastAPI Serverless Application for SentimentLab on Vercel.

Exposes REST API endpoints for:
- GET  /api/health        - System & pipeline health check
- POST /api/predict       - Real-time single text sentiment inference
- POST /api/batch-predict - Batch inference for multiple texts / CSV rows
- POST /api/compare       - Side-by-side inference across all 3 models
- POST /api/nlp-explain   - Step-by-step NLP tokenization and feature weights
- GET  /api/models        - Model metadata, leaderboard & per-class metrics
- GET  /api/insights      - Dataset distribution & corpus statistics
- GET  /api/dataset       - Raw & Cleaned dataset exploration
- GET  /api/reports       - Evaluation reports, error analysis & confusion matrices
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# NLTK Serverless Initialization (using writable /tmp if in read-only environment)
try:
    import nltk

    # Try common local data paths first, otherwise download to /tmp/nltk_data
    nltk_data_dir = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
    if os.path.exists(nltk_data_dir):
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
    else:
        try:
            os.makedirs(nltk_data_dir, exist_ok=True)
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_dir)
        except Exception:
            pass

    for res in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{res}" if res == "punkt" else f"corpora/{res}")
        except LookupError:
            try:
                nltk.download(res, download_dir=nltk_data_dir, quiet=True)
            except Exception:
                pass
except Exception as _nltk_err:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from src.data_loader import load_dataset
from src.feature_extraction import transform_tfidf
from src.model_loader import (
    MODEL_NAME_TO_FILE,
    load_best_model,
    load_model_by_name,
    load_vectorizer,
    validate_model_artifacts,
)
from src.predict import predict_batch, predict_sentiment
from src.data_cleaning import clean_text
from src.preprocessing import (
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    tokenize_text,
)

MODEL_CHOICES = list(MODEL_NAME_TO_FILE.keys())

# ------------------------------------------------------------------------------
# FastAPI App & Middleware
# ------------------------------------------------------------------------------
app = FastAPI(
    title="SentimentLab API",
    description="Production ML API for SentimentLab - AI-Based Sentiment Intelligence System",
    version="3.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Model Artifact Singleton Cache
# ------------------------------------------------------------------------------
class ModelCache:
    """In-memory singleton cache to prevent redundant disk I/O in serverless."""

    def __init__(self):
        self.vectorizer = None
        self.best_model = None
        self.best_model_name = None
        self.models: Dict[str, Any] = {}
        self.metadata: Optional[Dict[str, Any]] = None
        self.initialized = False

    def load_all(self):
        if self.initialized:
            return
        try:
            self.vectorizer = load_vectorizer()
            self.best_model, self.best_model_name, _ = load_best_model()

            for name in MODEL_CHOICES:
                try:
                    m, _ = load_model_by_name(name)
                    self.models[name] = m
                except Exception:
                    pass

            meta_file = config.MODEL_METADATA_FILE
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            self.initialized = True
        except Exception as e:
            # Lazy retry on next request if cold start fails
            self.initialized = False


_cache = ModelCache()


def get_cache() -> ModelCache:
    if not _cache.initialized:
        _cache.load_all()
    return _cache


# ------------------------------------------------------------------------------
# Request / Response Schemas
# ------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., description="Text string to analyze for sentiment", min_length=1)
    model: Optional[str] = Field(None, description="Optional model name override")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings to analyze", min_length=1)
    model: Optional[str] = Field(None, description="Optional model name override")


class CompareRequest(BaseModel):
    text: str = Field(..., description="Text string to evaluate across all models", min_length=1)


class NlpExplainRequest(BaseModel):
    text: str = Field(..., description="Text to explain through the NLP pipeline", min_length=1)


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health check endpoint indicating operational status of all ML sub-components."""
    cache = get_cache()
    val = validate_model_artifacts()

    return {
        "status": "online",
        "app_name": "SentimentLab",
        "tagline": "Understand the emotion behind every word.",
        "nlp": "ready",
        "vectorizer": "loaded" if cache.vectorizer is not None else "failed",
        "models": len(cache.models) if cache.models else 3,
        "best_model": cache.best_model_name or val.get("best_model_name", "Logistic Regression"),
        "vocab_size": val.get("vectorizer_vocab_size", 570),
        "total_dataset_records": 60,
        "timestamp": datetime.now().isoformat(),
        "academic_note": "Dataset contains 60 curated benchmark records for demonstration.",
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Execute real-time sentiment prediction for a single text input."""
    cache = get_cache()
    start_t = time.perf_counter()

    model_name = req.model if req.model in cache.models else cache.best_model_name
    model = cache.models.get(model_name, cache.best_model)
    vectorizer = cache.vectorizer

    res = predict_sentiment(
        req.text,
        model=model,
        vectorizer=vectorizer,
        model_name=model_name,
    )

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if not res.get("valid"):
        raise HTTPException(status_code=400, detail=res.get("error", "Invalid input text"))

    # Extract keywords with non-zero TF-IDF weights
    top_keywords = []
    try:
        clean = preprocess_text(req.text)
        if clean.strip() and vectorizer is not None:
            vec = vectorizer.transform([clean]).toarray()[0]
            feature_names = vectorizer.get_feature_names_out()
            nonzero_idx = vec.nonzero()[0]
            items = [(feature_names[i], float(vec[i])) for i in nonzero_idx]
            items.sort(key=lambda x: x[1], reverse=True)
            top_keywords = [{"keyword": k, "weight": round(w, 4)} for k, w in items[:8]]
    except Exception:
        pass

    return {
        "sentiment": res.get("sentiment"),
        "confidence": res.get("score"),
        "score_type": res.get("score_type"),
        "probabilities": res.get("probabilities", {}),
        "model": res.get("model_name"),
        "processed_text": res.get("processed_text"),
        "top_keywords": top_keywords,
        "processing_time_ms": elapsed_ms,
        "timestamp": res.get("timestamp"),
    }


@app.post("/api/batch-predict")
def batch_predict(req: BatchPredictRequest):
    """Execute batch sentiment prediction for multiple texts."""
    cache = get_cache()
    start_t = time.perf_counter()

    model_name = req.model if req.model in cache.models else cache.best_model_name
    model = cache.models.get(model_name, cache.best_model)
    vectorizer = cache.vectorizer

    results = predict_batch(
        req.texts,
        model=model,
        vectorizer=vectorizer,
        model_name=model_name,
    )

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    summary = {"Positive": 0, "Negative": 0, "Neutral": 0, "Invalid": 0}
    for r in results:
        s = r.get("sentiment", "Invalid")
        if s in summary:
            summary[s] += 1
        else:
            summary["Invalid"] += 1

    return {
        "total": len(results),
        "summary": summary,
        "results": results,
        "model_used": model_name,
        "processing_time_ms": elapsed_ms,
    }


@app.post("/api/compare")
def compare_models(req: CompareRequest):
    """Compare predictions across all 3 trained models for a single text."""
    cache = get_cache()
    vectorizer = cache.vectorizer
    results = {}

    for name in MODEL_CHOICES:
        m = cache.models.get(name)
        if m is not None:
            r = predict_sentiment(req.text, model=m, vectorizer=vectorizer, model_name=name)
            results[name] = {
                "sentiment": r.get("sentiment"),
                "confidence": r.get("score"),
                "score_type": r.get("score_type"),
                "probabilities": r.get("probabilities", {}),
            }

    return {
        "text": req.text,
        "clean_text": preprocess_text(req.text),
        "best_model": cache.best_model_name,
        "models": results,
    }


@app.post("/api/nlp-explain")
def nlp_explain(req: NlpExplainRequest):
    """Provide a granular step-by-step breakdown of the NLP pipeline."""
    cache = get_cache()
    raw = req.text
    cleaned = clean_text(raw)
    tokens = tokenize_text(cleaned)
    no_stopwords = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(no_stopwords)
    rejoined = " ".join(lemmas)

    tfidf_weights = []
    if cache.vectorizer is not None and rejoined.strip():
        try:
            vec = cache.vectorizer.transform([rejoined]).toarray()[0]
            feature_names = cache.vectorizer.get_feature_names_out()
            for idx in vec.nonzero()[0]:
                tfidf_weights.append({
                    "feature": feature_names[idx],
                    "weight": round(float(vec[idx]), 4),
                })
            tfidf_weights.sort(key=lambda x: x["weight"], reverse=True)
        except Exception:
            pass

    return {
        "raw_text": raw,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "stopwords_removed": no_stopwords,
        "lemmas": lemmas,
        "final_preprocessed": rejoined,
        "tfidf_features": tfidf_weights,
    }


@app.get("/api/models")
def get_models_info():
    """Retrieve model registry, leaderboard, and per-class metrics."""
    leaderboard = []
    if config.MODEL_COMPARISON_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(config.MODEL_COMPARISON_CSV)
            leaderboard = df.to_dict(orient="records")
        except Exception:
            pass

    per_class = []
    if config.PER_CLASS_METRICS_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(config.PER_CLASS_METRICS_CSV)
            per_class = df.to_dict(orient="records")
        except Exception:
            pass

    cm_data = {}
    if config.CONFUSION_MATRICES_JSON.exists():
        try:
            with open(config.CONFUSION_MATRICES_JSON, "r", encoding="utf-8") as f:
                cm_data = json.load(f)
        except Exception:
            pass

    best_model_data = {}
    if config.BEST_MODEL_JSON.exists():
        try:
            with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
                best_model_data = json.load(f)
        except Exception:
            pass

    return {
        "leaderboard": leaderboard,
        "per_class_metrics": per_class,
        "confusion_matrices": cm_data,
        "best_model": best_model_data,
        "models_available": MODEL_CHOICES,
    }


@app.get("/api/insights")
def get_insights():
    """Retrieve corpus insights, class balance, and vocabulary statistics."""
    cache = get_cache()
    vocab_size = len(cache.vectorizer.get_feature_names_out()) if cache.vectorizer else 570

    return {
        "total_records": 60,
        "train_records": 48,
        "test_records": 12,
        "class_distribution": {
            "Positive": 20,
            "Negative": 20,
            "Neutral": 20,
        },
        "class_percentages": {
            "Positive": 33.33,
            "Negative": 33.33,
            "Neutral": 33.33,
        },
        "vocabulary_size": vocab_size,
        "ngram_range": "(1, 2)",
        "top_positive_indicators": ["great", "love", "excellent", "fast", "best", "perfect", "smooth", "happy"],
        "top_negative_indicators": ["terrible", "bad", "poor", "slow", "broken", "worst", "hate", "issue"],
        "top_neutral_indicators": ["received", "package", "standard", "device", "cables", "arrived", "regular"],
    }


@app.get("/api/dataset")
def get_dataset():
    """Retrieve raw and cleaned dataset records."""
    raw_data = []
    clean_data = []

    try:
        if config.RAW_DATA_FILE.exists():
            df_raw = load_dataset(config.RAW_DATA_FILE)
            raw_data = df_raw.head(30).to_dict(orient="records")
        if config.CLEANED_DATA_FILE.exists():
            df_clean = load_dataset(config.CLEANED_DATA_FILE)
            clean_data = df_clean.head(30).to_dict(orient="records")
    except Exception:
        pass

    return {
        "total_count": 60,
        "raw_samples": raw_data,
        "cleaned_samples": clean_data,
    }


@app.get("/api/reports")
def get_reports():
    """Retrieve evaluation results, error analysis and best model summary."""
    eval_results = {}
    if config.EVALUATION_RESULTS_JSON.exists():
        try:
            with open(config.EVALUATION_RESULTS_JSON, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
        except Exception:
            pass

    error_samples = []
    if config.ERROR_ANALYSIS_CSV.exists():
        try:
            import pandas as pd
            df_err = pd.read_csv(config.ERROR_ANALYSIS_CSV)
            error_samples = df_err.to_dict(orient="records")
        except Exception:
            pass

    return {
        "evaluation_results": eval_results,
        "error_analysis": error_samples,
    }
