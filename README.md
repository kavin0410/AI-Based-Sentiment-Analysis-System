<div align="center">

# 🧠 SentimentLab
### *"Understand the emotion behind every word."*

**AI-Based Sentiment Analysis System Using NLP & Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)](https://nextjs.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green?style=flat-square)](https://nltk.org)
[![Tests](https://img.shields.io/badge/Tests-126%20Passing-brightgreen?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📋 Project Information

| Item | Details |
|------|---------|
| **Project Title** | SentimentLab — AI-Based Sentiment Analysis System |
| **Tagline** | *"Understand the emotion behind every word."* |
| **Candidate Name(s)** | To be filled |
| **Domain** | Natural Language Processing (NLP) / Machine Learning / Artificial Intelligence |
| **Dataset Name** | SentimentLab Demo Dataset (Custom Curated) |
| **Dataset Source** | Custom curated benchmark dataset |
| **Dataset Records** | 60 records (20 Positive · 20 Negative · 20 Neutral) |
| **Input Features** | Raw text string (free-form natural language) → TF-IDF feature vector (570 features) |
| **Target Variable** | `sentiment` — Categorical: Positive, Negative, Neutral |
| **Sentiment Classes** | Positive · Negative · Neutral |
| **Raw Dataset File** | [`data/raw/demo_dataset.csv`](data/raw/demo_dataset.csv) |
| **Cleaned Dataset File** | [`data/processed/cleaned_dataset.csv`](data/processed/cleaned_dataset.csv) |
| **Source Code** | Included — [`src/`](src/) |
| **Jupyter Notebook** | Included — [`notebooks/sentiment_analysis.ipynb`](notebooks/sentiment_analysis.ipynb) |
| **Model Files** | Included — [`models/`](models/) |
| **Application / Dashboard** | SentimentLab (Streamlit + Next.js + FastAPI) |
| **Screenshots** | [`screenshots/`](screenshots/) |
| **Documentation** | [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) |
| **GitHub Repository** | [https://github.com/kavin0410/AI-Based-Sentiment-Analysis-System](https://github.com/kavin0410/AI-Based-Sentiment-Analysis-System) |

---

## 📦 Included Project Components

| Component | File / Folder | Status |
|-----------|--------------|--------|
| **Source Code** | `src/` (8 Python modules) | ✅ Included |
| **Configuration** | `config.py` | ✅ Included |
| **Jupyter Notebook** | `notebooks/sentiment_analysis.ipynb` | ✅ Included |
| **Logistic Regression Model** | `models/logistic_regression.pkl` | ✅ Included |
| **Multinomial Naive Bayes Model** | `models/naive_bayes.pkl` | ✅ Included |
| **Linear SVM Model** | `models/linear_svm.pkl` | ✅ Included |
| **TF-IDF Vectorizer** | `models/tfidf_vectorizer.pkl` | ✅ Included |
| **Model Metadata** | `models/model_metadata.json` | ✅ Included |
| **Best Model Selection** | `results/best_model.json` | ✅ Included |
| **Evaluation Results** | `results/evaluation_results.json` | ✅ Included |
| **Per-Class Metrics** | `results/per_class_metrics.csv` | ✅ Included |
| **Error Analysis** | `results/error_analysis.csv` | ✅ Included |
| **Streamlit Application** | `app/app.py` + `app/components/` | ✅ Included |
| **Next.js Frontend** | `pages/index.tsx` | ✅ Included |
| **FastAPI Backend** | `api/index.py` | ✅ Included |
| **Test Suite** | `tests/` (126 tests) | ✅ Included |
| **Documentation** | `docs/PROJECT_REPORT.md` | ✅ Included |
| **Screenshots** | `screenshots/` | To be added |

---

## 🎯 Project Overview

SentimentLab is a complete, end-to-end **AI-based sentiment analysis platform** built as an academic minor project in Natural Language Processing and Machine Learning.

The system classifies any free-form text into one of three sentiment polarities:

- 🟢 **Positive** — Satisfied, approving, or happy sentiment
- 🔴 **Negative** — Unsatisfied, critical, or unhappy sentiment
- 🔵 **Neutral** — Factual, balanced, or non-opinionated sentiment

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SentimentLab Architecture                       │
├─────────────┬────────────────────────┬────────────────────────────┤
│  DATA LAYER │   ML PIPELINE (Python) │   PRESENTATION LAYER       │
│─────────────│────────────────────────│────────────────────────────│
│ Raw Dataset │  Stage 1: Data Loading │  Streamlit App (local)     │
│  60 records │  Stage 2: NLP Cleaning │  Next.js Frontend (Vercel) │
│  2 columns  │  Stage 3: TF-IDF + ML  │  FastAPI REST API          │
│  text, sent │  Stage 4: Evaluation   │  9 Navigation Tabs         │
│             │  Stage 5: Prediction   │  Real-time Inference       │
└─────────────┴────────────────────────┴────────────────────────────┘
```

---

## 🔬 NLP Pipeline (12-Step Processing)

```
Raw Text Input
     │
     ▼
 1. Convert to lowercase
 2. Expand contractions  (don't → do not, can't → cannot)
 3. Remove HTML tags
 4. Remove URLs
 5. Remove punctuation & special characters
 6. Collapse extra whitespace
 7. NLTK Tokenization        (split into word tokens)
 8. Stopword Removal         (preserve negations: not, no, never...)
 9. WordNet Lemmatization    (running → run, better → good)
10. Rejoin to clean string
11. TF-IDF Vectorization     (570 unigram + bigram features)
12. Model Inference          → Positive / Negative / Neutral
```

---

## 🧠 Machine Learning Models

| Model | Algorithm | Library Class | Hyperparameters |
|-------|-----------|--------------|-----------------|
| **Logistic Regression** | Linear Classifier | `sklearn.linear_model.LogisticRegression` | `max_iter=1000, solver=lbfgs, random_state=42` |
| **Multinomial Naive Bayes** | Probabilistic | `sklearn.naive_bayes.MultinomialNB` | `alpha=1.0` |
| **Linear SVM** | Support Vector Machine | `sklearn.svm.LinearSVC` | `max_iter=2000, dual=auto, random_state=42` |

### Feature Engineering

| Parameter | Value |
|-----------|-------|
| Vectorizer | TF-IDF (Term Frequency–Inverse Document Frequency) |
| N-Gram Range | (1, 2) — Unigrams and Bigrams |
| Max Features | 5,000 (actual vocabulary: **570 features**) |
| Sublinear TF | Enabled |
| Train / Test Split | 80% / 20% (48 train · 12 test, stratified) |
| Random State | 42 |

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| **Dataset Name** | SentimentLab Demo Dataset |
| **Source** | Custom curated benchmark dataset |
| **Total Records** | 60 |
| **Class Balance** | Perfectly balanced — 20 per class (33.33% each) |
| **Training Records** | 48 (80%) |
| **Test Records** | 12 (20%) |
| **Input Column** | `text` — free-form natural language string |
| **Target Column** | `sentiment` — `Positive`, `Negative`, `Neutral` |
| **Raw File** | `data/raw/demo_dataset.csv` |
| **Cleaned File** | `data/processed/cleaned_dataset.csv` |
| **Cleaned Column** | `clean_text` — preprocessed lemmatized tokens |

> ⚠️ **Academic Notice**: This is a demonstration dataset with 60 curated benchmark records. Evaluation metrics reflect performance on this small dataset and are intended for academic learning purposes only.

---

## 🖥️ Application & Dashboard

### SentimentLab Interface — 9 Navigation Tabs

| Tab | Icon | Description |
|-----|------|-------------|
| **Overview** | `○` | Hero banner, platform metrics, quick analyzer, architecture flow |
| **Analyze** | `⊹` | Real-time single-text inference with NLP breakdown, batch CSV analysis |
| **Insights** | `◈` | Corpus distribution, vocabulary analytics, N-gram indicators |
| **History** | `◷` | Session prediction audit log with filtering, search, and CSV export |
| **NLP Explorer** | `◎` | Interactive step-by-step NLP pipeline inspection with TF-IDF weights |
| **Model Intelligence** | `◉` | Leaderboard comparison, per-class metrics, confusion matrices |
| **Data** | `▣` | 60-record raw and cleaned dataset explorer |
| **Reports** | `▤` | Evaluation summary, misclassification error analysis |
| **Settings** | `⚙` | Live API health check, runtime specifications |

### Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **ML Engine** | Scikit-Learn | ≥ 1.3.0 |
| **NLP Preprocessing** | NLTK | ≥ 3.8.1 |
| **Feature Extraction** | TF-IDF (Scikit-Learn) | — |
| **Data Processing** | Pandas, NumPy | ≥ 2.0, ≥ 1.24 |
| **Serialization** | Joblib | ≥ 1.3.0 |
| **Local UI** | Streamlit | ≥ 1.28.0 |
| **Production Frontend** | Next.js 14 + React 18 + TypeScript | 14.2.x |
| **Production API** | FastAPI + Uvicorn | ≥ 0.100.0 |
| **Styling** | Tailwind CSS | ≥ 3.4 |
| **Deployment Target** | Vercel Serverless | — |
| **Testing** | Pytest + FastAPI TestClient | ≥ 7.4.0 |
| **Language** | Python 3.9+ | — |

---

## ⚙️ Installation Instructions

### Prerequisites
- Python 3.9 or higher
- Node.js 18 or higher (for the Next.js frontend)
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/kavin0410/AI-Based-Sentiment-Analysis-System.git
cd AI-Based-Sentiment-Analysis-System
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Install Node.js Dependencies (for Next.js frontend)

```bash
npm install
```

---

## 🚀 How to Run

### Option A — Streamlit Application (Local)

```bash
streamlit run app/app.py
```

Opens at: `http://localhost:8501`

### Option B — Full-Stack (Next.js + FastAPI, for Vercel-style dev)

**Terminal 1 — FastAPI Backend:**
```bash
uvicorn api.index:app --reload --port 8000
```

**Terminal 2 — Next.js Frontend:**
```bash
npm run dev
```

Frontend at: `http://localhost:3000`  
API docs at: `http://localhost:8000/api/docs`

### Option C — Jupyter Notebook

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

### Run Tests

```bash
pytest tests/ -v
# Expected: 126 passed
```

---

## 🌐 API Reference

Base URL (local): `http://localhost:8000`  
Base URL (Vercel): `https://<your-deployment>.vercel.app`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System health check |
| `POST` | `/api/predict` | Single text sentiment inference |
| `POST` | `/api/batch-predict` | Batch inference for multiple texts |
| `POST` | `/api/compare` | Compare all 3 models on one text |
| `POST` | `/api/nlp-explain` | Step-by-step NLP pipeline breakdown |
| `GET` | `/api/models` | Model leaderboard and per-class metrics |
| `GET` | `/api/insights` | Corpus distribution and vocabulary stats |
| `GET` | `/api/dataset` | Dataset records (raw and cleaned) |
| `GET` | `/api/reports` | Evaluation results and error analysis |

### Example — Predict Endpoint

**Request:**
```json
POST /api/predict
{
  "text": "I absolutely love this product! Exceptional quality."
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.84,
  "score_type": "probability",
  "probabilities": {
    "Positive": 0.84,
    "Neutral": 0.10,
    "Negative": 0.06
  },
  "model": "Logistic Regression",
  "top_keywords": [
    {"keyword": "love", "weight": 0.4821},
    {"keyword": "exceptional", "weight": 0.3914}
  ],
  "processing_time_ms": 12.4
}
```

---

## 🗂️ Project Structure

```
AI-Based-Sentiment-Analysis-System/
│
├── api/
│   └── index.py                      # FastAPI serverless backend (9 endpoints)
│
├── app/
│   ├── app.py                        # Streamlit application entry point
│   ├── components/                   # 9 UI component modules
│   │   ├── overview.py
│   │   ├── analyze.py
│   │   ├── intelligence.py
│   │   ├── history.py
│   │   ├── nlp_engine.py
│   │   ├── model_lab.py
│   │   ├── data_quality.py
│   │   ├── reports.py
│   │   └── system.py
│   └── static/
│       └── custom.css                # Premium light AI SaaS design system
│
├── data/
│   ├── raw/
│   │   └── demo_dataset.csv          # 60-record raw dataset (text, sentiment)
│   └── processed/
│       └── cleaned_dataset.csv       # NLP-cleaned dataset (text, sentiment, clean_text)
│
├── docs/
│   ├── PROJECT_REPORT.md             # Full technical project report
│   └── FINAL_STATUS.md              # Implementation status document
│
├── models/
│   ├── logistic_regression.pkl       # Trained Logistic Regression (14.7KB)
│   ├── naive_bayes.pkl               # Trained Multinomial Naive Bayes (27.6KB)
│   ├── linear_svm.pkl                # Trained Linear SVM (14.2KB)
│   ├── tfidf_vectorizer.pkl          # Fitted TF-IDF Vectorizer (23.1KB)
│   └── model_metadata.json           # Training configuration metadata
│
├── notebooks/
│   └── sentiment_analysis.ipynb     # Complete Jupyter analysis notebook
│
├── pages/
│   ├── _app.tsx                      # Next.js App wrapper
│   ├── _document.tsx                 # HTML document
│   └── index.tsx                     # SentimentLab full Next.js UI (9 tabs)
│
├── results/
│   ├── best_model.json               # Best model selection metadata
│   ├── evaluation_results.json       # Full evaluation metrics
│   ├── model_comparison.csv          # Model leaderboard table
│   ├── per_class_metrics.csv         # Per-class P / R / F1 breakdown
│   └── error_analysis.csv            # Misclassification analysis
│
├── screenshots/                      # UI screenshots (to be added)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset loading & validation
│   ├── data_cleaning.py              # Text cleaning pipeline
│   ├── preprocessing.py              # NLTK NLP preprocessing
│   ├── feature_extraction.py         # TF-IDF vectorization
│   ├── train_model.py                # Model training
│   ├── evaluate_model.py             # Model evaluation & metrics
│   ├── model_loader.py               # Artifact loading & caching
│   └── predict.py                    # Real-time prediction engine
│
├── styles/
│   └── globals.css                   # Tailwind CSS + global styles
│
├── tests/                            # 126-test automated test suite
│   ├── test_stage1_data.py
│   ├── test_stage2_nlp.py
│   ├── test_stage3_models.py
│   ├── test_stage4_evaluation.py
│   ├── test_stage5_prediction.py
│   ├── test_stage6_experience.py
│   └── test_api.py                   # FastAPI integration tests (12 tests)
│
├── config.py                         # Central configuration constants
├── requirements.txt                  # Python dependencies
├── package.json                      # Node.js dependencies (Next.js)
├── next.config.js                    # Next.js + API rewrite config
├── vercel.json                       # Vercel deployment configuration
├── tailwind.config.js                # Tailwind CSS configuration
├── tsconfig.json                     # TypeScript configuration
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This document
```

---

## 🧪 Test Coverage

```
tests/
├── test_stage1_data.py        — Data loading, validation, dataset integrity
├── test_stage2_nlp.py         — NLP preprocessing, tokenization, lemmatization
├── test_stage3_models.py      — TF-IDF, model training, artifact persistence
├── test_stage4_evaluation.py  — Metrics, confusion matrix, best model selection
├── test_stage5_prediction.py  — Prediction engine, input validation, batch predict
├── test_stage6_experience.py  — End-to-end application integration
└── test_api.py                — FastAPI endpoint integration (12 endpoints)

Total: 126 tests — All Passing ✅
```

---

## ⚠️ Limitations

1. **Small Dataset**: The 60-record demo dataset is for academic learning, not production.
2. **Evaluation Metrics**: Results reflect the small test set (12 samples). Real-world accuracy requires a significantly larger corpus.
3. **Domain Specificity**: The NLP pipeline is optimized for product/service review language.
4. **Language**: English-only text processing.
5. **No Deep Learning**: Intentionally uses classical ML to demonstrate foundational concepts.
6. **Negation Handling**: Limited to keyword preservation; complex negation chains are not fully modeled.

---

## 🔮 Future Enhancements

1. Expand dataset to 10,000+ samples from real-world review corpora (Amazon, Yelp, IMDb)
2. Add deep learning models: LSTM, BERT fine-tuning, RoBERTa
3. Implement multilingual sentiment support
4. Add real-time streaming prediction with WebSocket
5. Integrate live social media API feeds (Twitter/X, Reddit)
6. Add aspect-based sentiment analysis (ABSA)
7. Implement active learning loop for continuous model improvement

---

## 📚 Documentation

| Document | Location |
|----------|---------|
| Full Technical Report | [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) |
| API Reference | `http://localhost:8000/api/docs` (FastAPI Swagger UI) |
| NLP Pipeline | Section 10 of Project Report |
| Model Evaluation | Section 15 of Project Report |
| Future Scope | Section 20 of Project Report |

---

## 🔗 Repository

**GitHub:** [https://github.com/kavin0410/AI-Based-Sentiment-Analysis-System](https://github.com/kavin0410/AI-Based-Sentiment-Analysis-System)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🎓 Academic Notice

> This project is an academic minor project submitted for evaluation in the domain of **Natural Language Processing and Machine Learning**. The dataset contains **60 curated benchmark records** for demonstration purposes. Evaluation metrics are presented transparently without fabrication. This project does not claim production-grade performance.

---

<div align="center">
<strong>SentimentLab</strong> · "Understand the emotion behind every word." · Built with ❤️ using Python, NLTK, Scikit-Learn, FastAPI & Next.js
</div>
