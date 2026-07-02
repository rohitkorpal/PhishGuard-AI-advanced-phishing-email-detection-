# PhishGuard AI: Advanced Phishing Detection System

This repository contains a complete, end-to-end B.Tech AIML semester project that detects phishing/spam emails using Natural Language Processing (NLP) and Machine Learning techniques.

## 🛡️ Project Overview
Phishing emails are one of the most common vectors for cyberattacks. **PhishGuard AI** demonstrates how text classification algorithms can analyze text structure, patterns, and vocabulary to accurately distinguish between **Spam (Phishing)** and **Ham (Legitimate)** emails.

Using the provided dataset `enron_spam_data.csv`, the project achieves **99.13% accuracy** and **99.10% F1-score** with a Support Vector Machine (LinearSVC) model.

---

## 📁 Repository Structure
```text
├── enron_spam_data.csv      # Primary dataset (Enron Email Dataset)
├── requirements.txt         # Project dependencies
├── app.py                   # Streamlit web application
├── README.md                # Project documentation
├── Project_Report.md        # Comprehensive project report
├── train.py                 # Script to retrain the SVM model
├── generate_plots.py        # Script to generate visualizations
├── phishing_email_detection.ipynb  # Interactive Jupyter Notebook
├── models/
│   ├── best_model.pkl       # Serialized best-performing ML model (SVM)
│   └── tfidf_vectorizer.pkl # Serialized TF-IDF Vectorizer
└── visualizations/          # Generated evaluation graphs
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

### 4. Download NLTK Datasets
The project requires the `stopwords`, `wordnet`, `punkt`, and `omw-1.4` corpora from NLTK. They are downloaded automatically on first run, but you can also download them manually using Python:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

---

## 🚀 Execution Guide

### 1. Running the Jupyter Notebook
Open the notebook in your Jupyter environment:
```bash
jupyter notebook phishing_email_detection.ipynb
```
Run all cells to step through the text preprocessing pipelines, EDA, model training, and performance comparisons.

### 2. Launching the Streamlit Web Application
To start the interactive web application:
```bash
streamlit run app.py
```
This will spin up a local development server and open the web interface in your default browser (usually at `http://localhost:8501`).

---

## 📊 Summary of Model Performance (TF-IDF Features)

| Machine Learning Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **99.13%** | **98.68%** | **99.52%** | **99.10%** |
| **Logistic Regression** | 98.92% | 98.28% | 99.49% | 98.88% |
| **Multinomial Naive Bayes** | 98.85% | 98.90% | 98.70% | 98.80% |
| **Random Forest** | 98.57% | 98.13% | 98.91% | 98.52% |
| **Decision Tree** | 95.80% | 96.19% | 95.01% | 95.60% |

*The best model was chosen based on its **F1-score**, which balances classification accuracy on both minority (Spam) and majority (Ham) classes.*


