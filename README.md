# PhishGuard AI: Advanced Phishing Detection System

This repository contains a complete, end-to-end B.Tech AIML semester project that detects phishing/spam emails using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 🛡️ Project Overview
Phishing emails are one of the most common vectors for cyberattacks. **PhishGuard AI** demonstrates how text classification algorithms can analyze text structure, semantic patterns, and vocabulary to accurately distinguish between **Spam (Phishing)** and **Ham (Legitimate)** emails.

Using the provided dataset `dataset/enron_spam_data.csv` (30,494 unique emails), the project compares traditional Machine Learning and Deep Learning architectures. The baseline classifier **Support Vector Machine (LinearSVC)** achieves **99.13% accuracy** and **99.10% F1-score** on unseen test data.

---

## ✨ Core Features of the Web Application

1.  **📊 Dataset Insights (Tab 1)**: Visualizes the class balance (52% Ham / 48% Spam) and frequently used terms via side-by-side comparative wordclouds.
2.  **📈 Model Stats & Performance Visualizations (Tab 2)**: Displays interactive metric explainers, confusion matrix heatmaps with text interpretation, and a quantitative comparison table of the 5 classifiers.
3.  **🛡️ Unified Dual-Model Verification (Tab 3)**:
    *   Runs **Support Vector Machine (LinearSVC)** and **BERT (Contextual Embeddings)** simultaneously on the input email.
    *   Displays a **Consensus Safety Verdict** banner at the top (🚨 Critical Phishing, ⚠️ Suspicious Activity, or ✅ Safe Email).
    *   Renders side-by-side metrics panels comparing the output, confidence rating, and processing parameters of both models.
    *   **📧 Advanced Sender Header Audit**: Evaluates sender display names against actual email domain paths to flag Display Name Spoofing, Free Webmail Corporate Abuse, Homoglyph-Aware Levenshtein Typosquatting (e.g., `g00gle.com` or `rnicrosoft.com`), and Reply-To domain redirect attacks.
    *   **🔗 Hybrid Hyperlink Scanner**: Extracts email hyperlinks and evaluates them via (a) lexical heuristics (HTTP checks, obfuscated shorteners, raw IP hostnames), and (b) a GPU-trained **XGBoost URL Classifier** (trained on 651k URLs).
    *   **📄 Download Forensic Report**: Generates and downloads a formal PDF security audit report containing text verdicts, sender metadata verification logs, and link scan details.

---

## 📁 Repository Structure
```text
├── dataset/
│   ├── enron_spam_data.csv    # Primary dataset (Enron Email Dataset)
│   └── malicious_phish.csv    # Secondary dataset (651k Malicious URLs Dataset)
├── requirements.txt           # Project dependencies
├── app.py                     # Streamlit web application (includes SVM, BERT, URL pipelines)
├── predict_bert_demo.py       # Offline demo script to test local BERT pipeline
├── README.md                  # Project documentation
├── Project_Report.md          # Project report (IEEE Conference Paper style)
├── train.py                   # Script to train and compare the 5 ML models
├── train_url_model.py         # Script to train the XGBoost URL classifier
├── generate_plots.py          # Script to generate Enron evaluation graphs
├── phishing_email_detection.ipynb  # Interactive Jupyter Notebook
├── models/
│   ├── best_model.pkl         # Serialized best-performing ML model (SVM)
│   ├── tfidf_vectorizer.pkl   # Serialized TF-IDF Vectorizer
│   ├── url_classifier.pkl     # Serialized XGBoost URL classifier
│   └── url_vectorizer.pkl     # Serialized URL char-level TF-IDF Vectorizer
└── visualizations/            # Generated evaluation graphs
    ├── enron_class_distribution.png
    ├── enron_confusion_matrix.png
    └── enron_wordclouds.png
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Enable Local Deep Learning (BERT) Support (Optional)
To run the local offline BERT model inside the app and execute the demo script, install the deep learning dependencies:
```bash
pip install transformers torch
```
*(If these packages are not installed, the Streamlit app will gracefully fallback to the SVM classifier and prompt you on how to unlock the BERT feature).*

---

## 🚀 Execution Guide

### 1. Launching the Web Application
To start the Streamlit web application:
```bash
streamlit run app.py
```
This opens the browser dashboard at `http://localhost:8501`. Paste your email into Tab 3 to run the dual-analysis engine.

### 2. Running the Offline BERT Demo
To test the offline Hugging Face BERT model on a sample service upgrade phishing email:
```bash
python predict_bert_demo.py
```

### 3. Running the Jupyter Notebook
Open the notebook to explore the modeling phase:
```bash
jupyter notebook phishing_email_detection.ipynb
```

---

## 📊 Summary of Model Performance (TF-IDF Features)

The following table summarizes the performance of the 5 classifiers trained on the Enron Email dataset:

| Machine Learning Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **99.13%** | **98.68%** | **99.52%** | **99.10%** |
| **Logistic Regression** | 98.92% | 98.28% | 99.49% | 98.88% |
| **Multinomial Naive Bayes** | 98.85% | 98.90% | 98.70% | 98.80% |
| **Random Forest** | 98.57% | 98.13% | 98.91% | 98.52% |
| **Decision Tree** | 95.80% | 96.19% | 95.01% | 95.60% |

*The best model was chosen based on its **F1-score**, which balances classification accuracy on both minority (Spam) and majority (Ham) classes.*

---

## 🔗 Malicious URL Classifier Performance (XGBoost)

For link-level checking, the app loads an **XGBoost Classifier** trained on **641,119 unique URLs** with domain-disjoint splitting (80% train domains / 20% test domains) to prevent leakage. It achieves a validation accuracy of **95.41%**:

| Class Label | Precision | Recall | F1-Score | Validation Support |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 96% | 99% | 97% | 32,955 |
| **Phishing** | 90% | 79% | 84% | 7,219 |
| **Defacement** | 98% | 97% | 97% | 8,483 |
| **Malware** | 98% | 86% | 91% | 1,341 |
| **Overall Accuracy** | | | **95.41%** | 49,998 |
