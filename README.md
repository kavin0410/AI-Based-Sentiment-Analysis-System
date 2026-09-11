# AI Sentiment Intelligence
### AI-Based Sentiment Analysis System Using Machine Learning

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)
![pytest](https://img.shields.io/badge/pytest-7.4%2B-lightgrey.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Project Status](https://img.shields.io/badge/Status-Stage%207%20Complete-brightgreen.svg)

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Machine Learning Models](#-machine-learning-models)
- [NLP Preprocessing Pipeline](#-nlp-preprocessing-pipeline)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation Instructions](#-installation-instructions)
- [How to Run](#-how-to-run)
- [Evaluation Results](#-evaluation-results)
- [Project Pipeline Stages](#-project-pipeline-stages)
- [Testing](#-testing)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Academic Project Note](#-academic-project-note)
- [License](#-license)

---

## 🌟 Project Overview

**AI Sentiment Intelligence** is a complete end-to-end machine learning system for automatically analyzing and classifying human sentiment in unstructured text. It classifies text into three polarities:

- **Positive** — expressions of satisfaction, happiness, approval, or enthusiasm
- **Negative** — expressions of dissatisfaction, frustration, criticism, or hostility
- **Neutral** — factual statements, objective reporting, or queries without emotional polarity

Built with emphasis on clean software engineering, reproducibility, and academic rigor, this project implements a complete 7-stage pipeline from raw data ingestion through to a polished interactive Streamlit web application with real-time prediction.

> **Academic Transparency:** This is an academic minor project with a **60-record dataset (48 train / 12 test)**. Model performance is intentionally transparent — results reflect the real, unmodified evaluation on this small dataset. Performance figures are **not fabricated**.

---

## 🎯 Problem Statement

The exponential growth of digital platforms — social media, e-commerce, review aggregators, customer portals — generates immense volumes of unstructured opinion data. Understanding public sentiment is vital for:

1. **Consumer Experience & Brand Reputation**: Detecting negative feedback early to resolve complaints and safeguard customer trust.
2. **Product Feedback & Market Intelligence**: Identifying user sentiment regarding specific features to aid data-driven product development.
3. **Information Overload**: Manual inspection of thousands of reviews is labor-intensive, error-prone, and economically infeasible.

**The Challenge:** Human language is inherently ambiguous, filled with slang, informal grammar, abbreviations, and domain-specific nuances.

**The Solution:** An automated, scalable, and explainable machine learning system that standardizes noisy text, extracts discriminating n-gram features, and delivers transparent sentiment predictions with confidence scores.

---

## 🎯 Objectives

- **Develop a Robust NLP Preprocessing Pipeline**: Clean and normalize raw text through URL removal, case normalization, contraction expansion, tokenization, stopword removal, and lemmatization with negation preservation.
- **Extract High-Quality Features**: Convert textual tokens into sparse numerical matrices using TF-IDF with unigram and bigram representations (570 features).
- **Train and Compare Multiple ML Classifiers**: Build and benchmark Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
- **Comprehensive Evaluation**: Measure performance using Accuracy, Precision, Recall, Weighted F1-Score, and Confusion Matrices.
- **Automatic Best Model Selection**: Dynamically select the best performing model by weighted F1-score without hardcoding.
- **Deliver an Interactive Web Application**: Streamlit interface with real-time inference, NLP pipeline inspection, session history, and CSV export.
- **Ensure Academic & Engineering Rigor**: Clean PEP 8 code, modular design, full reproducibility, and comprehensive documentation.

---

## ✨ Key Features

- **⚡ Real-Time Sentiment Prediction**: Instant classification into Positive, Negative, or Neutral with confidence/decision scores.
- **⚖️ Automatic Best Model Selection**: Dynamic selection of the best model by weighted F1 — no hardcoding.
- **🧹 12-Step NLP Preprocessing Pipeline**: Contraction expansion, negation preservation, URL/HTML removal, tokenization, stopword filtering, lemmatization.
- **📊 TF-IDF Feature Extraction**: 570 features, `ngram_range=(1,2)`, `sublinear_tf=True`.
- **📈 Performance Intelligence Dashboard**: Confusion matrices, per-model metrics, error analysis, classification reports.
- **🔬 Interactive NLP Pipeline Simulator**: Step-by-step visualization of preprocessing transformation.
- **📜 Prediction History & Export**: Session-level history (last 50 predictions) with CSV download.
- **🛡️ Data Quality Audit**: Dataset audit tab with raw vs. cleaned preview and quality statistics.
- **🧪 Comprehensive Test Suite**: 114+ automated tests covering all pipeline stages.
- **📋 Academic Documentation**: VIVA Q&A guide, project report, final status, architecture documentation.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression
- **Basis**: Linear model estimating class membership probability via logistic/softmax functions.
- **Strengths**: Outstanding for high-dimensional sparse TF-IDF vectors; interpretable; well-calibrated probabilities.
- **Config**: `solver='lbfgs'`, `max_iter=1000`, `C=1.0`, `multi_class='auto'`, `random_state=42`

### 2. Multinomial Naive Bayes
- **Basis**: Probabilistic classifier applying Bayes' Theorem with conditional feature independence on word count distributions.
- **Strengths**: Classic NLP baseline; very fast training; suitable for small datasets.
- **Config**: `alpha=1.0` (Laplace smoothing)

### 3. Linear Support Vector Machine (LinearSVC)
- **Basis**: Maximum-margin classifier constructing optimal separating hyperplanes in high-dimensional vector spaces.
- **Strengths**: Effective in high-dimensional sparse spaces; robust against overfitting via margin maximization.
- **Config**: `C=1.0`, `max_iter=1000`, `random_state=42`
- **Note**: Uses `decision_function()` scores (not `predict_proba()`), labeled as "Decision Score" in the UI.

---

## 🔄 NLP Preprocessing Pipeline

Raw text is transformed through a 12-step pipeline:

```
[Raw Text Input]
  Step 1:  Lowercase conversion
  Step 2:  URL & HTML removal
  Step 3:  Contraction expansion (e.g., "can't" -> "cannot")
  Step 4:  User mention & special character removal
  Step 5:  Negation preservation (e.g., "not good" -> "not_good")
  Step 6:  Punctuation normalization
  Step 7:  Number removal
  Step 8:  Tokenization (NLTK word_tokenize)
  Step 9:  Stopword removal (negation tokens preserved)
  Step 10: Short token filtering (len < 2)
  Step 11: Lemmatization (WordNetLemmatizer)
  Step 12: Reconstruction into clean token string
[TF-IDF Vectorization -> 570-feature sparse matrix]
```

---

## 🛠 Technology Stack

| Category | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| Language | Python | 3.10+ | Core programming language |
| Data Manipulation | pandas | 2.0+ | Data loading, wrangling, CSV export |
| Numerical Processing | numpy | 1.24+ | Array and vector operations |
| Machine Learning | scikit-learn | 1.3+ | Classifiers, TF-IDF, metrics |
| NLP Tooling | NLTK | 3.8+ | Tokenization, stopwords, WordNet |
| Web Interface | Streamlit | 1.30+ | Interactive dashboard and prediction UI |
| Data Visualization | matplotlib | 3.7+ | Confusion matrix and metric plots |
| Statistical Visuals | seaborn | 0.12+ | Heatmaps, distribution aesthetics |
| Model Persistence | joblib | 1.3+ | Serialization of trained models |
| Testing | pytest | 7.4+ | Automated unit and integration tests |
| Prototyping | Jupyter Notebook | 7.0+ | EDA and pipeline experimentation |

---

## 🏗 System Architecture

```
[USER LAYER]
  Streamlit Web Dashboard (8-Tab UI) / Jupyter Notebook
       |
[APPLICATION LAYER]
  app/app.py  +  app/components/ (8 modular page renderers)
       |
[PREDICTION ENGINE LAYER]
  src/predict.py  +  src/model_loader.py
  PredictionHistoryManager  +  validate_input_text()
       |
[MODEL ARTIFACTS LAYER]
  models/logistic_regression.pkl  |  models/naive_bayes.pkl
  models/linear_svm.pkl  |  models/tfidf_vectorizer.pkl
  results/best_model.json  (auto-selected: Logistic Regression)
       |
[MODEL TRAINING & EVALUATION LAYER]
  src/train_model.py  +  src/evaluate_model.py
  Accuracy, Precision, Recall, F1, Confusion Matrices
       |
[FEATURE EXTRACTION LAYER]
  src/feature_extraction.py
  TF-IDF 570 features, ngram_range=(1,2), sublinear_tf=True
       |
[PREPROCESSING LAYER]
  src/preprocessing.py - 12-step NLP pipeline
  Contraction expansion, negation preservation, lemmatization
       |
[DATA LAYER]
  src/data_loader.py  +  src/data_cleaning.py
  data/raw/ --> data/processed/ (60 records, 48/12 split)
```

---

## 📁 Project Structure

```
AI-Sentiment-Intelligence/
├── data/
│   ├── raw/                           # Original dataset (60 records)
│   └── processed/                     # Cleaned data and train/test splits
├── notebooks/
│   └── sentiment_analysis.ipynb       # Full EDA and pipeline notebook
├── src/                               # Core modular Python source package
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── preprocessing.py               # 12-step NLP pipeline
│   ├── feature_extraction.py          # TF-IDF vectorization
│   ├── train_model.py                 # LR, MNB, LinearSVC training
│   ├── evaluate_model.py              # Metrics and best model selection
│   ├── model_loader.py                # Dynamic artifact loader
│   └── predict.py                     # Real-time prediction engine
├── models/                            # Serialized .pkl artifacts
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── linear_svm.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json
├── results/                           # Evaluation outputs
│   ├── evaluation_results.csv
│   ├── evaluation_results.json
│   ├── best_model.json
│   └── figures/                       # Confusion matrix heatmaps
├── app/
│   ├── app.py                         # Streamlit entry point (8-tab navigation)
│   └── components/                    # Modular tab renderers
│       ├── overview.py
│       ├── analyze.py
│       ├── intelligence.py
│       ├── model_lab.py
│       ├── nlp_engine.py
│       ├── data_quality.py
│       ├── reports.py
│       └── architecture.py
├── tests/                             # Automated test suite (114+ tests)
│   ├── test_foundation.py             # Stage 1 - config, paths, structure (24 tests)
│   ├── test_stage2_nlp.py             # Stage 2 - NLP pipeline (22 tests)
│   ├── test_stage3_models.py          # Stage 3 - model artifacts (15 tests)
│   ├── test_stage4_evaluation.py      # Stage 4 - evaluation results (27 tests)
│   ├── test_stage5_prediction.py      # Stage 5 - prediction engine (21 tests)
│   └── test_stage6_experience.py      # Stage 6 - Streamlit app (5 tests)
├── docs/
│   ├── VIVA_GUIDE.md                  # 20 Q&A viva preparation guide
│   ├── PROJECT_REPORT.md              # Full academic project report
│   ├── FINAL_STATUS.md                # Final project completion summary
│   └── screenshots/                   # UI screenshots (manual capture)
├── run_stage2.py                      # Stage 2 pipeline runner
├── run_stage3.py                      # Stage 3 pipeline runner
├── run_stage4.py                      # Stage 4 evaluation runner
├── run_stage5.py                      # Stage 5 prediction validator
├── config.py                          # Centralized configuration
├── requirements.txt                   # Python package dependencies
├── .env.example                       # Environment variable template
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT License
└── README.md                          # This file
```

---

## 📋 Dataset

| Property | Value |
| :--- | :--- |
| Total Records | 60 |
| Training Samples | 48 (80%) |
| Test Samples | 12 (20%) |
| Classes | Positive, Negative, Neutral |
| Class Distribution | Balanced - 20 per class |
| Split Strategy | Stratified, random_state=42, test_size=0.20 |
| TF-IDF Features | 570 |
| N-gram Range | (1, 2) - unigrams and bigrams |

The dataset is intentionally small as this is an academic minor project demonstrating the complete ML pipeline architecture, not a production-scale system.

---

## ⚙️ Installation Instructions

### 1. Clone or Open the Repository
```bash
git clone https://github.com/your-username/AI-Sentiment-Intelligence.git
cd AI-Sentiment-Intelligence
```

### 2. Create and Activate a Virtual Environment
Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. NLTK Resources
The preprocessing module auto-downloads NLTK resources via `ensure_nltk_resources()` on first run.
To download manually:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Run Training Pipeline (if models not yet present)
```bash
python run_stage2.py
python run_stage3.py
python run_stage4.py
python run_stage5.py
```

---

## 🚀 How to Run

### Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Opens at http://localhost:8501

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

### Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📊 Evaluation Results

> **Important:** All metrics below are **real, unmodified results** from evaluating trained models on the held-out test set (12 samples). No values are fabricated or adjusted.

### Model Comparison Table

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 (Weighted) | Test Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **33.33%** | **37.30%** | **33.33%** | **32.76%** | 12 |
| Multinomial Naive Bayes | 25.00% | 25.00% | 25.00% | 22.22% | 12 |
| Linear SVM | 33.33% | 37.30% | 33.33% | 32.76% | 12 |

**Best Model:** Logistic Regression (selected by weighted F1-score = 0.3276)

### Why Performance Is Low (and Honest)
- Only **12 test samples** — one wrong prediction shifts accuracy by ~8%.
- **60 total records** is a demonstration dataset, not production scale.
- Random chance baseline = 33.33% (3 equal classes). The models meet but don't exceed this on the small test set.
- This is a known, documented limitation — not a pipeline error.

---

## 📐 Project Pipeline Stages

| Stage | Description | Status |
| :--- | :--- | :---: |
| Stage 1 | Project Foundation - architecture, config, directory structure, base test suite | Complete |
| Stage 2 | Dataset & NLP Engine - data loading, cleaning, 12-step NLP, EDA | Complete |
| Stage 3 | Feature Engineering & ML Training - TF-IDF, LR, MNB, LinearSVC serialization | Complete |
| Stage 4 | Model Evaluation - metrics, confusion matrices, error analysis, auto best-model selection | Complete |
| Stage 5 | Real-Time Prediction Engine - predict.py, model_loader.py, history manager | Complete |
| Stage 6 | Sentiment Intelligence Experience - 8-tab Streamlit UI, dark theme, modular components | Complete |
| Stage 7 | Final Engineering, Documentation & Submission | Complete |

---

## 🧪 Testing

```bash
pytest tests/ -v
```

| Test File | Area | Tests |
| :--- | :--- | :---: |
| test_foundation.py | Config, paths, structure, imports | 24 |
| test_stage2_nlp.py | NLP pipeline, cleaning, tokenization, lemmatization | 22 |
| test_stage3_models.py | Model artifacts, TF-IDF, serialization | 15 |
| test_stage4_evaluation.py | Evaluation results, best model JSON, metric ranges | 27 |
| test_stage5_prediction.py | Prediction engine, history manager, input validation | 21 |
| test_stage6_experience.py | Streamlit app structure, component imports | 5 |
| **Total** | | **114+** |

---

## ⚠️ Limitations

1. **Small Dataset**: 60 records is a demonstration dataset. Performance does not generalize to production use.
2. **Low Test Accuracy**: ~33% reflects the dataset size, not a pipeline defect.
3. **No Deep Learning**: Only classical ML models (LR, NB, SVM) - no BERT, LSTM, or transformers.
4. **English Only**: All preprocessing is designed for English text.
5. **LinearSVC No Probabilities**: Decision scores are used instead of probabilities for LinearSVC.
6. **Single-Text UI**: Current Streamlit UI processes one text at a time; batch upload is a future feature.

---

## 🔮 Future Enhancements

1. **Larger Dataset**: Source 5,000-50,000 labelled records for robust generalization.
2. **Transformer Models**: Fine-tune BERT or DistilBERT for contextual semantic understanding.
3. **Aspect-Based Sentiment (ABSA)**: Associate sentiments with specific product aspects.
4. **Multilingual Support**: Extend pipeline for multiple languages.
5. **REST API**: Package inference as FastAPI service with Docker.
6. **Model Explainability (XAI)**: Integrate LIME or SHAP for token-level explanations.
7. **Batch File Upload**: Add CSV/TXT batch upload to Streamlit UI.
8. **Real-Time Data Streaming**: Integrate live social media APIs for real-time monitoring.

---

## 🎓 Academic Project Note

This repository is an **Academic Minor Project** submitted as part of the undergraduate computer science and engineering curriculum.

- **Deadline**: 11 September 2026
- **Focus Areas**: Natural Language Processing, Machine Learning Pipelines, Supervised Classification, Model Evaluation.
- **Evaluation & Viva Readiness**: All metrics are real, all design decisions are documented in `docs/VIVA_GUIDE.md`.

Metrics intentionally reflect real performance on a small academic dataset. No values have been fabricated. This is by design.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

*AI Sentiment Intelligence - Academic Minor Project | Built with Python, scikit-learn, NLTK, and Streamlit*
