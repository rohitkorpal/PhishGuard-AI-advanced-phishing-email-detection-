import streamlit as st
import pickle
import os
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sys
import subprocess

# Try to import transformers and torch; auto-install if missing
try:
    import transformers
    import torch
    from transformers import pipeline
    HAS_BERT = True
except ImportError:
    print("Deep learning libraries (transformers, torch) not found. Installing automatically...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
        import transformers
        import torch
        from transformers import pipeline
        HAS_BERT = True
        print("Deep learning libraries installed successfully!")
    except Exception as e:
        print(f"Auto-installation of deep learning libraries failed: {e}")
        HAS_BERT = False

# Setup NLTK paths
nltk_data_dir = os.path.expanduser("~/nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Page Configuration
st.set_page_config(
    page_title="PhishGuard AI: Advanced Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) for premium look
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #560bad;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #7209b7;
        box-shadow: 0px 4px 15px rgba(114, 9, 183, 0.4);
        transform: translateY(-2px);
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
    }
    .spam-box {
        background-color: #ffe5ec;
        color: #d90429;
        border: 2px solid #ef233c;
    }
    .ham-box {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
    .info-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Preprocessing Functions
@st.cache_resource
def load_nlp_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except Exception:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('omw-1.4')
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    return lemmatizer, stop_words

lemmatizer, stop_words = load_nlp_resources()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and Lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(cleaned_tokens)

# Load Models
@st.cache_resource
def load_models():
    model_path = os.path.join("models", "best_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

model, vectorizer = load_models()

@st.cache_resource
def load_bert_model():
    if HAS_BERT:
        try:
            return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
        except Exception:
            return None
    return None

# Sidebar Setup
st.sidebar.markdown("<h2 style='color:#560bad;'>🛡️ PhishGuard AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
**PhishGuard AI** is an NLP-based project that analyzes incoming emails and classifies them as either **Ham (Legitimate)** or **Spam (Phishing)**.

### 📊 Model Info:
* **Best Model:** Support Vector Machine (LinearSVC)
* **Feature Extractor:** TF-IDF Vectorizer
*   **Accuracy:** 99.13%
*   **F1-Score:** 99.10%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Preprocessing Pipeline:")
st.sidebar.markdown("""
1. Text Lowercasing
2. HTML & URL removal
3. Number & Punctuation removal
4. Tokenization
5. NLTK Stopwords filtering
6. WordNet Lemmatization
""")

st.sidebar.markdown("---")

# Main Content
st.markdown("<div class='info-header'>🛡️ PhishGuard AI: Advanced Phishing Detection System</div>", unsafe_allow_html=True)

# Create 3 tabs
tab1, tab2, tab3 = st.tabs(["📊 Dataset Insights", "📈 Model Stats & Visualizations", "🛡️ Verify Email"])

with tab1:
    st.header("📊 Enron Email Dataset Insights")
    st.markdown("""
    The model is trained on the **Enron Email Dataset**, which contains real-world emails categorized into legitimate (Ham) and spam/phishing (Spam).
    This dataset provides a much more realistic learning environment than short-form SMS spam datasets.
    """)
    
    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Emails", "30,494")
    with col2:
        st.metric("Spam Emails (Phishing)", "14,584")
    with col3:
        st.metric("Ham Emails (Legitimate)", "15,910")
        
    st.markdown("### Class Distribution & Word Frequency")
    
    # Display class distribution image
    dist_image_path = os.path.join("visualizations", "enron_class_distribution.png")
    if os.path.exists(dist_image_path):
        col_dist1, col_dist2, col_dist3 = st.columns([1, 2, 1])
        with col_dist2:
            st.image(dist_image_path, caption="Enron Dataset Class Distribution", use_container_width=True)
        
    # Display WordClouds
    wc_image_path = os.path.join("visualizations", "enron_wordclouds.png")
    if os.path.exists(wc_image_path):
        col_wc1, col_wc2, col_wc3 = st.columns([1, 4, 1])
        with col_wc2:
            st.image(wc_image_path, caption="Frequently Used Words (Spam vs. Ham)", use_container_width=True)

with tab2:
    st.header("📈 Model Performance & Evaluation Metrics")
    st.markdown("""
    Below are the performance metrics, validation graphs, and classifier comparisons for the **Support Vector Machine (LinearSVC)** model trained on the Enron dataset.
    """)
    
    # Styled Callout Box
    st.markdown("""
    <div style='background-color: #f0f4f8; padding: 1.25rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 5px solid #560bad;'>
        <h4 style='color: #1e3a8a; margin: 0; font-family: sans-serif;'>🏆 Selected Classifier: Support Vector Machine (LinearSVC)</h4>
        <p style='color: #374151; font-size: 0.95rem; margin-top: 0.5rem; line-height: 1.4; font-family: sans-serif;'>
            LinearSVC operates by constructing an optimal hyperplane (decision boundary) that separates raw spam emails from legitimate ones in a high-dimensional TF-IDF feature space. By maximizing the margins between classes, it achieves highly robust generalization.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Training Accuracy", "99.99%")
    with col2:
        st.metric("Testing Accuracy", "99.13%")
    with col3:
        st.metric("Precision", "98.68%")
    with col4:
        st.metric("Recall", "99.52%")
    with col5:
        st.metric("F1-Score", "99.10%")
        
    # Interactive Metrics Explainer
    with st.expander("📚 What do these evaluation metrics actually mean?"):
        st.markdown("""
        *   **Training Accuracy (99.99%)**: The classification accuracy on the dataset the model learned from.
        *   **Testing Accuracy (99.13%)**: The classification accuracy on the unseen test dataset. The tiny difference (~0.86%) indicates a highly generalized, non-overfit model.
        *   **Precision (98.68%)**: Out of all emails the model flagged as *Spam*, how many were actually *Spam*. High precision means **fewer legitimate emails are blocked by mistake** (low False Positives).
        *   **Recall (99.52%)**: Out of all actual *Spam* emails received, how many did the model successfully catch. High recall means **fewer phishing attempts slip into the inbox** (low False Negatives).
        *   **F1-Score (99.10%)**: The harmonic mean of Precision and Recall. It is the gold-standard metric for evaluating balanced text classification.
        """)

    st.markdown("---")
    
    # Grid: Confusion Matrix Left, Details Right
    col_cm1, col_cm2 = st.columns([1.2, 1])
    
    with col_cm1:
        st.markdown("### 🎯 Confusion Matrix")
        cm_image_path = os.path.join("visualizations", "enron_confusion_matrix.png")
        if os.path.exists(cm_image_path):
            st.image(cm_image_path, caption="LinearSVC Confusion Matrix (Test Set)", use_container_width=True)
            
    with col_cm2:
        st.markdown("### 🔍 Matrix Interpretation")
        st.markdown("""
        The confusion matrix lists predictions on **6,099 unseen test emails**:
        
        *   ✅ **Ham correctly identified (True Negatives)**: **3,149** emails (~99.2%)
        *   ❌ **Ham flagged as Spam (False Positives)**: **26** emails (~0.8%) *(Minimal disruption)*
        *   ❌ **Spam missed (False Negatives)**: **14** emails (~0.5%) *(Extremely secure)*
        *   ✅ **Spam correctly blocked (True Positives)**: **2,910** emails (~99.5%)
        
        This shows that the model successfully balances high security (catching spam) with high usability (not blocking legitimate business emails).
        """)
        
    st.markdown("---")

    st.markdown("### 📊 Model Comparison (TF-IDF Features)")
    st.markdown("""
    Below is the comparison table showing how all 5 classifiers performed on the Enron test set:
    
    | Machine Learning Model | Accuracy | Precision | Recall | F1 Score |
    | :--- | :---: | :---: | :---: | :---: |
    | **Support Vector Machine (LinearSVC)** | **99.13%** | **98.68%** | **99.52%** | **99.10%** |
    | **Logistic Regression** | 98.92% | 98.28% | 99.49% | 98.88% |
    | **Multinomial Naive Bayes** | 98.85% | 98.90% | 98.70% | 98.80% |
    | **Random Forest** | 98.57% | 98.13% | 98.91% | 98.52% |
    | **Decision Tree** | 95.80% | 96.19% | 95.01% | 95.60% |
    """)
    
    # Explainable AI: Why SVM won
    with st.expander("❓ Why did the Support Vector Machine (LinearSVC) outperform other models?"):
        st.markdown("""
        1. **High-Dimensional Feature Space**: The TF-IDF extraction generated **227,132 features (n-grams)**. SVMs are mathematically designed to find the optimal separating hyperplane in high dimensions.
        2. **Linear Separability of Text**: In text classification, datasets are often linearly separable if the feature space is large enough. Models like SVM and Logistic Regression excel at drawing these linear boundaries.
        3. **Resistance to Overfitting**: By maximizing the margin between classes (Ham and Spam), SVM avoids overfitting on specific terms, maintaining a high testing performance (**99.13%**) close to its training score.
        """)

with tab3:
    st.header("🛡️ Verify Email Content")
    st.markdown("Enter the content of the email you wish to analyze in the text box below. The machine learning model will classify it and output the probability score.")
    
    if model is None or vectorizer is None:
        st.error("Error: Trained model files not found! Please run the training script to generate 'best_model.pkl' and 'tfidf_vectorizer.pkl' in the 'models' directory.")
    else:
        # Text input
        email_text = st.text_area("Email Content", height=200, placeholder="Paste your email text here...")

        if st.button("Predict / Analyze"):
            if email_text.strip() == "":
                st.warning("Please enter some text to classify.")
            else:
                # -----------------------------------------------------
                # PART 1: Run SVM (Traditional ML) Pipeline
                # -----------------------------------------------------
                cleaned = preprocess_text(email_text)
                vectorized = vectorizer.transform([cleaned])
                pred_svm = model.predict(vectorized)[0]
                
                # SVM Probability
                prob_svm = 0.5
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vectorized)[0]
                    prob_svm = probs[1]
                elif hasattr(model, "decision_function"):
                    decision_score = model.decision_function(vectorized)[0]
                    prob_svm = 1 / (1 + np.exp(-decision_score))
                
                # Apply rules override for SVM
                text_lower = email_text.lower()
                lottery_spam = "free" in text_lower and any(w in text_lower for w in ["won", "prize", "iphone", "claim"])
                
                threat_words = ["suspend", "suspension", "restrict", "restricted", "closure", "deactivate", "deactivation",
                                "unusual login", "security alert", "action required", "blocked", "compromised", "unauthorized"]
                soft_words = ["deadline", "validate", "validation", "upgrade", "uninterrupted", "enhancement"]
                action_words = ["verify", "verification", "confirm", "update", "restore", "portal", "link", "click here", "login"]
                link_cta = ["link", "url", "http", "click", "below", "button", "visit", "form", "portal", "website"]
                has_link_instruction = any(w in text_lower for w in link_cta)
                
                phishing_spam_urgent = any(w in text_lower for w in threat_words) and any(w in text_lower for w in action_words)
                phishing_spam_soft_svm = (
                    any(w in text_lower for w in soft_words) and 
                    any(w in text_lower for w in action_words) and 
                    has_link_instruction and 
                    pred_svm == 1
                )
                phishing_spam_svm = phishing_spam_urgent or phishing_spam_soft_svm
                
                if lottery_spam or phishing_spam_svm:
                    pred_svm = 1
                    prob_svm = max(prob_svm, 0.95)

                # -----------------------------------------------------
                # PART 2: Run BERT (Deep Learning) Pipeline
                # -----------------------------------------------------
                pred_bert = None
                prob_bert = None
                bert_pipeline = None
                
                if HAS_BERT:
                    with st.spinner("Loading BERT model weights... (This can take a few seconds on first run)"):
                        bert_pipeline = load_bert_model()
                    
                    if bert_pipeline is not None:
                        with st.spinner("Analyzing text using BERT..."):
                            truncated_text = email_text[:1000]
                            res = bert_pipeline(truncated_text)[0]
                            
                            pred_bert = 1 if res['label'] == 'LABEL_1' else 0
                            prob_bert = float(res['score'])
                            if pred_bert == 0:
                                prob_bert = 1.0 - prob_bert
                                
                            # Apply overrides to BERT
                            phishing_spam_soft_bert = (
                                any(w in text_lower for w in soft_words) and 
                                any(w in text_lower for w in action_words) and 
                                has_link_instruction and 
                                pred_bert == 1
                            )
                            phishing_spam_bert = phishing_spam_urgent or phishing_spam_soft_bert
                            if lottery_spam or phishing_spam_bert:
                                pred_bert = 1
                                prob_bert = max(prob_bert, 0.95)

                # -----------------------------------------------------
                # PART 3: Consensus Verdict Banner
                # -----------------------------------------------------
                st.markdown("### 🛡️ Unified Safety Verdict")
                if HAS_BERT and bert_pipeline is not None:
                    # Consensus between SVM and BERT
                    if pred_svm == 1 and pred_bert == 1:
                        st.error("🚨 **CRITICAL VERDICT: HIGH CONFIDENCE PHISHING ALERT** (Both traditional ML and deep learning models flagged this email as spam/phishing)")
                    elif pred_svm == 1 or pred_bert == 1:
                        st.warning("⚠️ **WARNING VERDICT: SUSPICIOUS ACTIVITY** (One of the models flagged this email. Caution is strongly advised)")
                    else:
                        st.success("✅ **SAFE VERDICT: LEGITIMATE EMAIL** (Both models verified this email as safe and legitimate)")
                else:
                    # SVM only verdict
                    if pred_svm == 1:
                        st.error("🚨 **VERDICT: PHISHING ALERT DETECTED** (The SVM Classifier flagged this email as spam/phishing)")
                    else:
                        st.success("✅ **VERDICT: LEGITIMATE EMAIL** (The SVM Classifier verified this email as safe)")

                # -----------------------------------------------------
                # PART 4: Side-by-Side Analysis Panels
                # -----------------------------------------------------
                col_l, col_r = st.columns(2)
                
                with col_l:
                    st.markdown("#### 📊 Traditional ML (LinearSVC)")
                    # Cleaned tokens view
                    with st.expander("🔍 See Preprocessed Text"):
                        st.write(f"**Cleaned Tokens:** `{cleaned}`")
                        
                    if pred_svm == 1:
                        st.markdown("<div class='result-box spam-box'>⚠️ SPAM DETECTED</div>", unsafe_allow_html=True)
                        st.metric("Spam Confidence Score", f"{prob_svm*100:.2f}%")
                        st.progress(float(prob_svm))
                    else:
                        st.markdown("<div class='result-box ham-box'>✅ LEGITIMATE (HAM)</div>", unsafe_allow_html=True)
                        st.metric("Legitimate Confidence Score", f"{(1 - prob_svm)*100:.2f}%")
                        st.progress(float(1 - prob_svm))
                        
                with col_r:
                    st.markdown("#### 🤖 Deep Learning (BERT)")
                    if not HAS_BERT:
                        st.info("💡 **BERT Model is Locked**")
                        st.warning("Hugging Face `transformers` and `torch` libraries are required to enable local BERT analysis.")
                        st.code("pip install transformers torch", language="bash")
                    elif bert_pipeline is None:
                        st.error("Failed to load BERT model weights from cache.")
                    else:
                        with st.expander("🔍 See BERT Details"):
                            st.write(f"**BERT Label:** `{res['label']}`")
                            st.write(f"**Raw BERT Score:** `{res['score']:.4f}`")
                            st.write("**Representation:** Dense Contextual Word Embeddings (768 dimensions)")
                            
                        if pred_bert == 1:
                            st.markdown("<div class='result-box spam-box'>⚠️ SPAM DETECTED</div>", unsafe_allow_html=True)
                            st.metric("Spam Confidence Score", f"{prob_bert*100:.2f}%")
                            st.progress(float(prob_bert))
                        else:
                            st.markdown("<div class='result-box ham-box'>✅ LEGITIMATE (HAM)</div>", unsafe_allow_html=True)
                            st.metric("Legitimate Confidence Score", f"{(1 - prob_bert)*100:.2f}%")
                            st.progress(float(1 - prob_bert))
                
st.markdown("---")