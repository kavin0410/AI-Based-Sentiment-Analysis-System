# 🎓 Viva Preparation & Oral Examination Guide

**Project Title:** AI Sentiment Intelligence: AI-Based Sentiment Analysis System Using Machine Learning  
**Target Deadline:** 11 September 2026  
**Classification Domain:** 3-Class Multiclass Sentiment Analysis (**Positive**, **Negative**, **Neutral**)

---

## 📌 Core Project Concepts & Objectives

### Q1: What is the primary objective of this project?
**A:** To design, evaluate, and deploy an end-to-end Machine Learning sentiment intelligence system that classifies input text into Positive, Negative, or Neutral sentiment categories using traditional NLP preprocessing, TF-IDF feature extraction, multi-model evaluation, and an interactive Streamlit web interface.

### Q2: Why is Sentiment Analysis important in real-world applications?
**A:** Sentiment analysis automates the extraction of human emotions and opinions from unstructured text (product reviews, social media posts, customer support tickets), allowing organizations to gain actionable intelligence, monitor brand reputation, and automate customer support routing.

### Q3: Why did you choose traditional ML (Logistic Regression, Naive Bayes, Linear SVM) over Deep Learning or Large Language Models (LLMs)?
**A:** Traditional ML models combined with TF-IDF offer high inference speed, minimal computational overhead (no GPUs required), full interpretability of feature weights, reproducible baseline performance, and deterministic deployment suitable for lightweight academic demonstration.

---

## 🔬 NLP Preprocessing & Feature Extraction

### Q4: What are the main steps in your NLP preprocessing pipeline?
**A:**
1. **Safe String Normalization & Lowercasing:** Converts text to lowercase to ensure uniform token matching.
2. **Noise Cleaning:** Strips URLs, HTML tags/entities, email addresses, and non-alphanumeric special characters.
3. **Contraction Expansion:** Expands words like *"don't"* → *"do not"* so negations survive stopword filtering.
4. **Tokenization:** Breaks text into individual word tokens using NLTK `word_tokenize`.
5. **Controlled Stopword Removal with Negation Preservation:** Removes uninformative stopwords while strictly retaining sentiment-critical negations (*not, no, never, neither, nor, without, against*).
6. **WordNet Lemmatization:** Reduces words to their dictionary root form (*running* → *run*, *better* → *good*).

### Q5: Why is Negation Preservation crucial in sentiment analysis?
**A:** Standard stopword lists strip negation words like *"not"* or *"no"*. Stripping *"not"* from *"not good"* leaves *"good"*, completely reversing the sentiment class from Negative to Positive. Preserving negations prevents sentiment inversion.

### Q6: What is the difference between Stemming and Lemmatization? Why did you select Lemmatization?
**A:** Stemming cuts off prefixes/suffixes using heuristic rules often producing non-words (e.g., *"studies"* → *"studi"*). Lemmatization uses morphological analysis and a dictionary (WordNet) to return valid canonical word roots (e.g., *"studies"* → *"study"*), yielding cleaner TF-IDF features.

### Q7: What is TF-IDF (Term Frequency-Inverse Document Frequency)?
**A:** TF-IDF quantifies word importance in a document relative to a corpus.
- **Term Frequency (TF):** Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** Penalizes terms that appear frequently across all documents (like common filler words), giving higher weight to distinctive sentiment-bearing words.

### Q8: What configuration parameters were used for your TF-IDF vectorizer?
**A:**
- `max_features=5000`: Caps the vocabulary size to the top 5,000 informative n-grams.
- `ngram_range=(1, 2)`: Captures both single words (unigrams like *"good"*) and two-word phrases (bigrams like *"not good"*).
- `sublinear_tf=True`: Replaces raw TF count $t$ with $1 + \log(t)$ to diminish the impact of repetitive terms.
- `min_df=1`: Includes terms appearing in at least 1 document.

---

## 🧠 Machine Learning Algorithms & Selection

### Q9: Explain how Logistic Regression works for multi-class sentiment analysis.
**A:** Logistic Regression uses the linear combination of TF-IDF feature weights passed through a softmax function to output calibrated class probability distributions for Positive, Negative, and Neutral.

### Q10: How does Multinomial Naive Bayes work for text classification?
**A:** Multinomial Naive Bayes applies Bayes' Theorem with the "naive" assumption of conditional independence between word features given the class label. It computes the posterior class probability based on term frequencies in the training corpus.

### Q11: How does Linear Support Vector Machine (LinearSVM) operate?
**A:** LinearSVM finds optimal hyperplanes in high-dimensional feature space that maximize the margin between sentiment classes using a linear decision boundary (`LinearSVC`).

### Q12: Why does LinearSVM not provide `predict_proba()` by default? How did you handle this?
**A:** `LinearSVC` optimizes a margin-based hinge loss rather than probabilistic log-loss, so it computes class decision boundary distances via `decision_function()` rather than probabilities. In our prediction engine, LinearSVM outputs are explicitly labeled as **Decision Scores** with user disclaimers, ensuring technical honesty.

### Q13: How is the "Best Model" selected automatically?
**A:**
1. **Primary Metric:** Weighted F1-score across test predictions.
2. **Secondary Metric (Tie-Breaker):** Overall Accuracy.
3. Selection executes programmatically in Stage 4 and saves metadata to `results/best_model.json`, which `src/model_loader.py` dynamically resolves during inference without hardcoding.

---

## 📊 Model Evaluation & Metrics

### Q14: What is the difference between Accuracy, Precision, Recall, and F1-Score?
**A:**
- **Accuracy:** Proportion of total correct predictions out of all test samples ($\frac{TP+TN}{Total}$).
- **Precision:** Proportion of true positive predictions out of all instances predicted as that class ($\frac{TP}{TP+FP}$). Measures prediction exactness.
- **Recall:** Proportion of true positive predictions out of all actual ground-truth instances of that class ($\frac{TP}{TP+FN}$). Measures completeness.
- **F1-Score:** The harmonic mean of Precision and Recall ($\frac{2 \cdot P \cdot R}{P + R}$). Balances precision and recall, especially for multi-class problems.

### Q15: Why is Weighted F1-Score preferred over Accuracy for evaluation?
**A:** Accuracy can be misleading if class distributions vary or when misclassification costs differ. Weighted F1-score averages per-class F1-scores weighted by class support, providing a reliable measure of model balance.

### Q16: What is a Confusion Matrix?
**A:** A $3 \times 3$ grid comparing ground-truth actual sentiment labels (rows) against model predicted labels (columns). It highlights specific misclassification patterns (e.g. Neutral samples misclassified as Negative).

---

## 🛡️ Data Leakage & Reproducibility

### Q17: How did you ensure zero data leakage in your evaluation pipeline?
**A:**
1. The 80/20 train/test split was executed *before* any feature extraction using fixed `random_state=42`.
2. The TF-IDF vectorizer was fitted exclusively on training text (`fit_transform()`).
3. Test set text was transformed using `.transform()` only (no refitting).
4. Models were evaluated strictly on unseen test predictions.

### Q18: Why is `random_state=42` used across the codebase?
**A:** To ensure complete experimental reproducibility. Setting fixed random seeds guarantees identical train/test splits and model initialization across runs and machines.

---

## ⚠️ Transparent Performance & Limitations

### Q19: Why is the current model evaluation accuracy (33.33%) and weighted F1-score (32.76%) relatively low?
**A:** The current demonstration dataset contains 60 total records (48 train / 12 test). With 12 test samples (4 per class), single misclassifications significantly impact percentage scores. The primary focus of this project stage was establishing a robust, leak-free end-to-end engineering pipeline.

### Q20: What are the primary limitations of the current system?
**A:**
- Small dataset size (60 records).
- Vocabulary limited to 570 TF-IDF features.
- Inability to capture complex context, sarcasm, or long-range semantics compared to Transformer models (BERT).

---

## 🚀 Quick Reference Summary Table for Oral Defense

| Concept | Project Implementation |
|---|---|
| **Supported Sentiments** | 3 Classes: Positive, Negative, Neutral |
| **Dataset Size** | 60 Records (48 Train / 12 Test - Stratified 80/20) |
| **TF-IDF Vocabulary** | 570 Features (Unigrams + Bigrams, Sublinear TF) |
| **Evaluated Models** | Logistic Regression, Multinomial Naive Bayes, Linear SVM |
| **Top Model Selected** | **Logistic Regression** (Weighted F1: 32.76%, Accuracy: 33.33%) |
| **Web Interface** | Streamlit 8-Tab Application (`app/app.py`) |
| **Test Suite** | 114 Unit Tests Passing (`pytest tests/`) |
