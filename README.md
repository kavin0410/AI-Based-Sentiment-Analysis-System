# AI Sentiment Intelligence
### AI-Based Sentiment Analysis System Using Machine Learning

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Project Status](https://img.shields.io/badge/Status-Stage%201%20Foundation-blueviolet.svg)

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Machine Learning Models](#-machine-learning-models)
- [NLP Preprocessing Pipeline](#-nlp-preprocessing-pipeline)
- [Technology Stack](#-technology-stack)
- [Planned System Architecture](#-planned-system-architecture)
- [Project Structure](#-project-structure)
- [Installation Instructions](#-installation-instructions)
- [How to Run](#-how-to-run)
- [Evaluation Results](#-evaluation-results)
- [Future Enhancements](#-future-enhancements)
- [Academic Project Note](#-academic-project-note)
- [License](#-license)

---

## 🌟 Project Overview

**AI Sentiment Intelligence** is an end-to-end machine learning system designed to automatically analyze, categorize, and interpret human sentiment embedded within unstructured text data. The system classifies textual content into three distinct sentiment polarities:

- **Positive** (expressions of satisfaction, happiness, approval, or enthusiasm)
- **Negative** (expressions of dissatisfaction, frustration, criticism, or hostility)
- **Neutral** (factual statements, objective reporting, or queries without emotional polarity)

Built with an emphasis on clean software engineering, reproducibility, and rigorous model evaluation, this project implements a complete pipeline spanning raw data ingestion, text preprocessing, TF-IDF vectorization, multi-model training, statistical evaluation, and an interactive Streamlit user interface.

---

## 🎯 Problem Statement

The exponential proliferation of digital platforms—such as social media, e-commerce storefronts, review aggregators, and customer service portals—has generated immense volumes of unstructured opinion data. Understanding public sentiment is vital for:

1. **Consumer Experience & Brand Reputation**: Organizations must detect negative feedback early to resolve complaints and safeguard customer trust.
2. **Product Feedback & Market Intelligence**: Identifying user sentiment regarding specific features aids data-driven product development.
3. **Information Overload**: Manual inspection of thousands of reviews or comments is labor-intensive, error-prone, subjective, and economically infeasible.

**The Challenge:** Human language is inherently ambiguous, filled with slang, informal grammar, abbreviations, punctuation variations, and domain-specific nuances.

**The Solution:** An automated, scalable, and explainable machine learning system capable of standardizing noisy text, extracting discriminating n-gram features, and delivering accurate sentiment predictions with confidence scores.

---

## 🎯 Objectives

- **Develop a Robust NLP Preprocessing Pipeline**: Clean and normalize raw text through URL removal, case normalization, regex noise filtering, tokenization, stopword removal, and lemmatization.
- **Extract High-Quality Features**: Convert textual tokens into sparse numerical matrices using Term Frequency-Inverse Document Frequency (TF-IDF) with unigram and bigram representations.
- **Train and Compare Multiple ML Classifiers**: Build and systematically benchmark three baseline and linear models: Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machine (Linear SVM).
- **Comprehensive Evaluation**: Measure performance using standard classification metrics including Accuracy, Precision, Recall, Macro/Weighted F1-Scores, and Confusion Matrices.
- **Deliver an Interactive Web Application**: Deploy a user-friendly Streamlit interface for single-text inference, batch file analysis, sentiment probability visualization, and query history.
- **Ensure Academic & Engineering Rigor**: Maintain clean PEP 8 compliant code, modular design, full reproducibility, and clear documentation suitable for academic evaluation and viva defense.

---

## ✨ Key Features

- **⚡ Real-Time Sentiment Prediction**: Instant classification of user-submitted text into Positive, Negative, or Neutral polarities.
- **⚖️ Multiple ML Model Comparison**: Side-by-side comparative analysis of Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
- **🧹 NLP Preprocessing Pipeline**: Clean, modular text standardization routines handles noise, contractions, special characters, and morphological inflections.
- **📊 TF-IDF Feature Extraction**: Sublinear term frequency scaling and n-gram ranges (1, 2) to capture local contextual patterns.
- **🎯 Confidence & Probability Scores**: Transparent probability distributions showing class-wise likelihood for each prediction.
- **📈 Analytics Dashboard**: Visual breakdown of sentiment proportions, text length distributions, and key vocabulary insights.
- **💻 Modern Streamlit UI**: Intuitive, responsive web application featuring clean styling, sentiment badges, and interactive widgets.
- **📜 Prediction History**: Session-level tracking of recent queries and classifications with export capability.

---

## 🤖 Machine Learning Models

The system evaluates three established supervised machine learning algorithms, each offering distinct theoretical advantages for natural language classification:

### 1. Logistic Regression
- **Theoretical Basis**: A linear model that estimates the probability of class membership using the logistic (sigmoid) and softmax functions over linear combinations of input features.
- **Why Chosen**: 
  - Outstanding baseline for high-dimensional, sparse TF-IDF text representations.
  - Highly interpretable coefficients that directly indicate which words correlate with positive or negative sentiment.
  - Fast training convergence and well-calibrated posterior probability estimates.

### 2. Multinomial Naive Bayes
- **Theoretical Basis**: A probabilistic classifier based on Bayes' Theorem under the assumption of conditional feature independence, adapted specifically for multinomially distributed word counts and frequencies.
- **Why Chosen**:
  - Classic NLP baseline renowned for strong performance on sparse word frequency matrices.
  - Extremely fast training and evaluation with low computational overhead.
  - Performs remarkably well on small to medium-sized text datasets without overfitting.

### 3. Linear Support Vector Machine (Linear SVM)
- **Theoretical Basis**: A maximum-margin classifier that constructs optimal separating hyperplanes in high-dimensional vector spaces to maximize the distance between classes.
- **Why Chosen**:
  - Exceptionally effective in high-dimensional sparse spaces where feature counts exceed or match sample counts.
  - Robust against overfitting due to margin maximization principles and $L_2$ regularization.
  - Historically delivers top-tier empirical accuracy on text categorization benchmarks.

---

## 🔄 NLP Preprocessing Pipeline

To transform noisy raw text into clean, representative tokens, the data passes through a multi-stage NLP pipeline:

```
[Raw Text Input]
       │
       ▼
1. Noise Cleaning & Regex Filtering (Strip URLs, HTML tags, user handles, punctuation, numbers)
       │
       ▼
2. Case Normalization (Convert all characters to lowercase)
       │
       ▼
3. Tokenization (Decompose text into individual word tokens using NLTK)
       │
       ▼
4. Stopword Removal (Filter domain-neutral stopwords while preserving negation context)
       │
       ▼
5. Lemmatization (Reduce words to base morphological roots using WordNetLemmatizer)
       │
       ▼
6. TF-IDF Vectorization (Compute Term Frequency-Inverse Document Frequency sparse matrix)
       │
       ▼
[Vectorized Feature Matrix (X)]
```

### Preprocessing Operations Detail:
1. **Cleaning**: Strips web links (`http(s)://...`), Twitter/social mentions (`@username`), non-alphanumeric artifacts, and extra whitespace.
2. **Normalization**: Lowercases strings to unify casing (`"Good"` and `"good"` map to identical representations).
3. **Tokenization**: Segment sentences into individual linguistic tokens using NLTK's word tokenization algorithms.
4. **Stopword Filtering**: Removes high-frequency linguistic glue words (e.g., `"the"`, `"is"`, `"at"`) that add negligible sentiment signal.
5. **Lemmatization**: Replaces inflected forms with base dictionary headwords (`"running"`, `"runs"` $\rightarrow$ `"run"`) via WordNet.
6. **Feature Extraction**: Constructs a numerical feature matrix using `TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)`.

---

## 🛠 Technology Stack

| Category | Technology / Library | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10+ | Core programming language |
| **Data Manipulation** | pandas | 2.0+ | Data loading, tabular wrangling, CSV operations |
| **Numerical Processing** | numpy | 1.24+ | Array manipulation and vector operations |
| **Machine Learning** | scikit-learn | 1.3+ | Classifiers, TF-IDF vectorizer, cross-validation, metrics |
| **NLP Tooling** | NLTK | 3.8+ | Tokenization, stopword lists, WordNet lemmatizer |
| **Web Interface** | Streamlit | 1.30+ | Interactive dashboard and real-time prediction UI |
| **Data Visualization** | matplotlib | 3.7+ | Metric plots, training curves, confusion matrix plotting |
| **Statistical Visuals** | seaborn | 0.12+ | Heatmaps and distribution aesthetics |
| **Model Persistence** | joblib | 1.3+ | Serialization of trained models and vectorizer state |
| **Prototyping** | Jupyter Notebook | 7.0+ | Exploratory analysis, pipeline experimentation |

---

## 🏗 Planned System Architecture

The architecture adheres to a clean layered design, separating data persistence, preprocessing, model execution, and user interaction:

```
┌──────────────────────────────────────────────────────────────────┐
│                           USER LAYER                             │
│       Streamlit Web Dashboard  /  Jupyter Notebook Interface     │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
│   Prediction Engine (predictor.py)  •  Analytics & Visualizer    │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                         INFERENCE LAYER                          │
│   Trained Models (models/*.joblib)  •  Fitted TF-IDF Vectorizer  │
└────────────────────────────────┬─────────────────────────────────┘
                                 ▲
                                 │ [Loads artifacts after training]
┌────────────────────────────────┴─────────────────────────────────┐
│                      MODEL TRAINING & EVAL                       │
│    model_trainer.py (Logistic Reg, Naive Bayes, Linear SVM)      │
│    evaluator.py (Accuracy, Precision, Recall, F1, Confusion Mtx) │
└────────────────────────────────┬─────────────────────────────────┘
                                 ▲
                                 │ [Feeds transformed features]
┌────────────────────────────────┴─────────────────────────────────┐
│                     FEATURE EXTRACTION LAYER                     │
│       feature_extractor.py (TF-IDF N-grams, Vocabulary Mapping)  │
└────────────────────────────────┬─────────────────────────────────┘
                                 ▲
                                 │ [Feeds cleaned tokens/text]
┌────────────────────────────────┴─────────────────────────────────┐
│                      PREPROCESSING LAYER                         │
│     preprocessor.py (Cleaning, Lowercase, Tokenize, Lemma)       │
└────────────────────────────────┬─────────────────────────────────┘
                                 ▲
                                 │ [Streams raw dataset rows]
┌────────────────────────────────┴─────────────────────────────────┐
│                          DATA LAYER                              │
│       data_loader.py (data/raw/ ───► data/processed/)            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
AI-Sentiment-Intelligence/
├── data/
│   ├── raw/                       # Original, immutable datasets
│   └── processed/                 # Cleaned, tokenized, preprocessed data splits
├── notebooks/
│   └── sentiment_analysis.ipynb   # Exploratory data analysis & prototyping notebook
├── src/                           # Core modular Python packages
│   ├── __init__.py                # Package initializer
│   ├── data_loader.py             # Dataset loading, validation, and split utilities
│   ├── preprocessor.py            # Text cleaning, tokenization, lemmatization logic
│   ├── feature_extractor.py       # TF-IDF vectorization and feature matrix builders
│   ├── model_trainer.py           # Training routines for LR, MNB, and Linear SVM
│   ├── evaluator.py               # Metric calculation, classification reports, confusion matrix
│   └── predictor.py               # Real-time single and batch prediction service
├── models/                        # Serialized .joblib artifacts (trained models & vectorizers)
├── results/                       # Generated outputs and evaluation artifacts
│   ├── figures/                   # Confusion matrix plots, charts, and visualizations
│   └── metrics/                   # Model performance summaries and evaluation JSON/CSV
├── app/
│   └── app.py                     # Interactive Streamlit web application
├── screenshots/                   # Application screenshots and demo captures
├── docs/                          # Academic documentation, reports, and architecture diagrams
├── tests/                         # Unit and integration test suites
│   ├── __init__.py
│   └── test_preprocessor.py       # Tests for data preprocessing and cleaning functions
├── config.py                      # Global configuration parameters, random seeds, paths
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git exclusion rules for artifacts and virtual environments
└── README.md                      # Project documentation and guide
```

---

## ⚙️ Installation Instructions

Follow these step-by-step instructions to set up the development environment locally on your workstation:

### 1. Clone or Open the Repository
```bash
git clone https://github.com/your-username/AI-Sentiment-Intelligence.git
cd AI-Sentiment-Intelligence
```

### 2. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Upgrade pip and Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpora
Run the following one-liner to download the requisite linguistic corpora (stopwords, WordNet lemmatizer data, and tokenizers):
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
```

---

## 🚀 How to Run

### 1. Run the Prototyping Notebook
To examine exploratory data analysis, pipeline steps, and model experimentation interactively:
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

### 2. Launch the Streamlit Web Application
To start the interactive sentiment prediction and dashboard interface:
```bash
streamlit run app/app.py
```
*The web interface will open automatically in your browser at `http://localhost:8501`.*

### 3. Run Unit Tests
To verify pipeline correctness and ensure preprocessing routines operate properly:
```bash
pytest tests/
```

---

## 📊 Evaluation Results

> [!IMPORTANT]
> **Stage 1 Status Notice:** Model evaluation metrics and comparison figures are **to be generated after model training**. In adherence to academic integrity principles, no simulated, estimated, or fabricated benchmark results are included prior to full pipeline execution.

Once model training completes, this section will report standardized empirical metrics evaluated on the held-out test split:

### Benchmark Comparison Table *(Placeholder)*

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Multinomial Naive Bayes** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Linear SVM** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

### Confusion Matrix Evaluation *(Placeholder)*
- *Confusion matrix heatmaps for each classifier will be exported to `results/figures/` upon evaluation.*
- Detailed classification reports with per-class metrics (Positive, Negative, Neutral) will be saved in `results/metrics/`.

---

## 🔮 Future Enhancements

1. **Deep Learning Architectures**: Implement Recurrent Neural Networks (LSTM, BiLSTM) and Gated Recurrent Units (GRU) to capture sequential linguistic dependencies.
2. **Transformer-Based Models**: Fine-tune state-of-the-art pretrained contextual language models (BERT, RoBERTa, DistilBERT) for enhanced semantic comprehension.
3. **Aspect-Based Sentiment Analysis (ABSA)**: Deconstruct reviews to associate sentiment polarities with specific entities or product aspects (e.g., battery life, customer service, price).
4. **Multilingual Sentiment Classification**: Expand the tokenization and feature extraction pipeline to support languages beyond English.
5. **Production REST API**: Package the inference service as a containerized FastAPI application with Docker for production scalability.
6. **Real-Time Data Streaming Ingestion**: Integrate live API pipelines (e.g., Reddit, YouTube comments) for real-time brand sentiment tracking.
7. **Model Explainability (XAI)**: Integrate LIME or SHAP to visually explain individual token contributions toward final sentiment decisions.

---

## 🎓 Academic Project Note

This repository constitutes an **Academic Minor Project** developed as part of the undergraduate computer science and engineering curriculum. 

- **Primary Academic Focus**: Natural Language Processing, Machine Learning Pipelines, Supervised Classification, and Model Evaluation.
- **Evaluation & Viva Readiness**: The repository is architected with clear modular separation, type hinting, and documented components to allow transparent review, reproducibility, and rigorous academic defense.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete terms and details.
