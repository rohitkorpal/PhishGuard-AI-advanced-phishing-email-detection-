# PhishGuard AI: Advanced Phishing Detection System

This repository contains a complete, end-to-end B.Tech AIML semester project that detects phishing/spam emails using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 🛡️ Project Overview
Phishing emails are one of the most common vectors for cyberattacks. **PhishGuard AI** demonstrates how text classification algorithms can analyze text structure, semantic patterns, and vocabulary to accurately distinguish between **Spam (Phishing)** and **Ham (Legitimate)** emails.

Using the provided dataset `dataset/enron_spam_data.csv` (30,494 unique emails), the project compares traditional Machine Learning and Deep Learning architectures. The baseline classifier **Support Vector Machine (LinearSVC)** achieves **99.13% accuracy** and **99.10% F1-score** on unseen test data.

---

## ✨ Core Features of the Web Application & Ecosystem

1.  **📊 Dataset Insights (Tab 1)**: Visualizes the class balance (52% Ham / 48% Spam) and frequently used terms via side-by-side comparative wordclouds.
2.  **📈 Model Stats & Performance Visualizations (Tab 2)**: Displays interactive metric explainers, confusion matrix heatmaps with text interpretation, and a quantitative comparison table of the 5 classifiers.
3.  **🛡️ Unified Dual-Model Verification (Tab 3)**:
    *   Runs **Support Vector Machine (LinearSVC)** and **BERT (Contextual Embeddings)** simultaneously on the input email.
    *   Displays a **Consensus Safety Verdict** banner at the top (🚨 Critical Phishing, ⚠️ Suspicious Activity, or ✅ Safe Email).
    *   Renders side-by-side metrics panels comparing the output, confidence rating, and processing parameters of both models.
    *   **📧 Advanced Sender Header Audit**: Evaluates sender display names against actual email domain paths to flag Display Name Spoofing, Free Webmail Corporate Abuse, Homoglyph-Aware Levenshtein Typosquatting (e.g., `g00gle.com` or `rnicrosoft.com`), and Reply-To domain redirect attacks.
    *   **🔗 Hybrid Hyperlink Scanner**: Extracts email hyperlinks and evaluates them via (a) lexical heuristics (HTTP checks, obfuscated shorteners, raw IP hostnames), and (b) a GPU-trained **XGBoost URL Classifier** (trained on 651k URLs).
    *   **📄 Download Forensic Report**: Generates and downloads a formal PDF security audit report containing text verdicts, sender metadata verification logs, and link scan details.
4.  **🔄 Adaptive Continuous Learning (Feedback Loop)**:
    *   Allows users (in both Streamlit and the Chrome Extension) to submit verification corrections.
    *   Corrections are appended to the feedback dataset. When submitted, the system triggers an asynchronous background retraining pipeline that re-fits the TF-IDF feature extractor and updates the SVM classification weights on disk.
    *   Uses **Zero-Day Local Threat Intelligence** (`models/local_intel.json`) to instantly block blacklisted sender domains and emails on subsequent scans, bypassing background training latency.
5.  **⚡ CUDA-GPU Acceleration**:
    *   Automatically detects local GPUs (`torch.cuda.is_available()`) to load the Hugging Face BERT classifier, accelerating deep learning text classification times.

---

## 📁 Repository Structure
```text
├── dataset/
│   ├── enron_spam_data.csv    # Primary dataset (Enron Email Dataset)
│   ├── malicious_phish.csv    # Secondary dataset (651k Malicious URLs Dataset)
│   └── feedback_data.csv      # Logged user classification corrections
├── chrome-extension/          # Manifest V3 Google Chrome extension directory
│   ├── manifest.json          # Configuration metadata and declarations
│   ├── popup.html             # Sleek glassmorphic popup UI
│   ├── popup.css              # Cyber-themed styling and micro-animations
│   ├── popup.js               # Event triggers, API integrations, and extraction
│   └── content.js             # Webmail page DOM extraction scripts
├── requirements.txt           # Project dependencies
├── app.py                     # Streamlit web application dashboard
├── api.py                     # FastAPI REST API backend (for the Chrome extension)
├── predict_bert_demo.py       # Offline demo script to test local BERT pipeline
├── README.md                  # Project documentation
├── Project_Report.md          # Project report (IEEE Conference Paper style)
├── LICENSE                    # Proprietary copyright protection notice
├── train.py                   # Script to train and compare the 5 ML models
├── train_url_model.py         # Script to train the XGBoost URL classifier
├── generate_plots.py          # Script to generate Enron evaluation graphs
├── phishing_email_detection.ipynb  # Interactive Jupyter Notebook
├── models/
│   ├── best_model.pkl         # Serialized best-performing ML model (SVM)
│   ├── tfidf_vectorizer.pkl   # Serialized TF-IDF Vectorizer
│   ├── url_classifier.pkl     # Serialized XGBoost URL classifier
│   ├── url_vectorizer.pkl     # Serialized URL char-level TF-IDF Vectorizer
│   └── local_intel.json       # Manually blacklisted emails and domains database
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

---

## 🔌 Chrome Extension Integration

PhishGuard AI includes a Google Chrome extension (Manifest V3) that allows real-time security auditing and NLP phishing classification directly inside your browser. It supports automated email body extraction and metadata scraping from webmail providers (Gmail and Microsoft Outlook) and can also run quick manual audits on link targets.

### 1. Architecture Overview
Because Chrome Extensions run in a JavaScript sandbox, the Python machine learning models (LinearSVC, XGBoost, BERT) cannot execute locally in the extension. Instead, the extension acts as a client that queries a local **FastAPI REST API Server** (`api.py`). The API runs the pre-processing pipelines, feeds inputs into the serialized model classifiers, conducts header integrity checks, and returns a unified consensus verdict.

```
[ Active Webmail Page ]
      │ (Inject DOM Scraper Script)
      ▼
[ PhishGuard Extension ] ───(HTTP POST)───► [ FastAPI Backend (api.py) ]
                                                    │ (Run ML Classifiers & Rules)
                                                    ▼
                                            [ best_model.pkl (SVM) ]
                                            [ url_classifier.pkl (XGBoost) ]
                                            [ BERT Pipeline (Hugging Face) ]
```

### 2. How to Run the API Backend
Before loading the Chrome extension, start the local API server:

1. Activate your virtual environment and ensure you have `fastapi` and `uvicorn` installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI application:
   ```bash
   python api.py
   ```
   The API will start locally at `http://127.0.0.1:8000`. You can verify it is running by visiting the root page in your browser.

### 3. How to Load the Chrome Extension
1. Open Google Chrome and navigate to the Extensions page: `chrome://extensions/`.
2. Toggle the **Developer mode** switch in the top-right corner to **ON**.
3. Click the **Load unpacked** button in the top-left corner.
4. Select the `chrome-extension` directory inside this repository.
5. The **PhishGuard AI** extension badge will appear in your Chrome toolbar. Click to open the popup.

### 4. Features & Usage
*   **API Status**: The extension automatically polls the FastAPI server. The status indicator at the top will turn green (`API Online`) when the backend is active.
*   **⚡ Extract & Scan Current Email**: Open any active email thread inside Gmail or Microsoft Outlook Web, open the extension, and click this button. The extension will scrape the sender name, email address, reply-to header, and full message body, send them to the API, and display a comprehensive security verdict.
*   **Manual Scan**: Paste any text block manually, customize sender headers, and click **Analyze Content**.
*   **Quick Link Scan**: Switch to the **Quick Link Scan** tab, paste a URL, and click **Inspect Hyperlink** to get an instant heuristic and XGBoost classification rating for that link.

