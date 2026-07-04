# PhishGuard AI: Advanced Phishing Email Detection System Using Natural Language Processing

**Rohit**  
*Department of Artificial Intelligence & Machine Learning (AIML)*  
*B.Tech Final Year Semester Project, Academic Year: 2026*  

---

### Abstract
Phishing remains one of the most persistent and dangerous cybersecurity threats, leading to significant financial losses and data breaches worldwide. Attackers exploit human vulnerabilities by sending deceptive emails that mimic legitimate communications, coercing victims into revealing sensitive credentials, installing malware, or authorizing unauthorized transactions. Traditional signature-based detection systems fall short against modern, dynamic phishing campaigns that utilize sophisticated social engineering tactics. This project presents an end-to-end Machine Learning (ML) and Natural Language Processing (NLP) framework, named **PhishGuard AI**, designed to identify and classify phishing emails automatically. Utilizing a public dataset of 33,716 emails (`dataset/enron_spam_data.csv`), we perform rigorous data cleaning and textual preprocessing—including HTML tag stripping, URL and number removal, tokenization, stopword removal, and lemmatization. For feature extraction, we evaluate and compare vectorization techniques: Bag of Words (CountVectorizer) and Term Frequency-Inverse Document Frequency (TF-IDF). Five distinct supervised learning algorithms are trained and compared: Logistic Regression, Multinomial Naive Bayes, Random Forest, Decision Tree, and Support Vector Machine (LinearSVC). Our experimental findings show that the Support Vector Machine (LinearSVC) model utilizing TF-IDF features achieves the best classification performance, boasting a validation accuracy of **99.13%** and an F1-score of **99.10%**. To transition this research into a practical utility, a lightweight web application is developed using Streamlit, permitting users to perform real-time security assessment of email content.

***Index Terms*—Phishing Detection, Natural Language Processing, Machine Learning, Support Vector Machines, TF-IDF, Cybersecurity.**

---

## I. INTRODUCTION

### A. Context
In the digital age, electronic mail (email) remains the cornerstone of professional and personal communication. However, its ubiquitous nature makes it an attractive vector for cybercriminals. According to global cybersecurity reports, phishing accounts for over 90% of successful cyberattacks and data breaches. Phishing is a form of social engineering where attackers impersonate trustworthy entities—such as banks, utilities, government bodies, or company executives—to manipulate individuals into performing specific actions.

### B. Motivation
Traditionally, email providers relied on blacklist filters, keyword matching, and IP reputation systems to block malicious communications. While effective against bulk, uncoordinated spam, these techniques are ineffective against targeted phishing attacks (spear-phishing) and zero-day exploits, where the email content contains no known signatures and originates from legitimate but compromised servers. Recent advancements in Natural Language Processing (NLP) and Machine Learning (ML) present a paradigm shift in cybersecurity. By framing phishing detection as a text classification problem, models can capture semantic structure, contextual anomalies, and linguistic features characteristic of phishing campaigns, regardless of changing domain names or IP addresses.

### C. Scope of the System
This project explores the design, development, and evaluation of an NLP-based phishing email detection system. The scope encompasses:
1. Parsing and cleaning a messy, real-world email dataset.
2. Formulating a robust text preprocessing pipeline to isolate core semantic tokens.
3. Conducting Exploratory Data Analysis (EDA) to understand vocabulary patterns, message lengths, and class balances.
4. Converting raw text into numerical matrices using Term Frequency-Inverse Document Frequency (TF-IDF) methodologies.
5. Training and hyperparameter-tuning five predictive models.
6. Systematically comparing performance using multi-dimensional evaluation metrics (Accuracy, Precision, Recall, F1-Score).
7. Deploying a functional prototype via Streamlit.

---

## II. PROBLEM FORMULATION & SYSTEM OBJECTIVES

### A. Problem Statement
The core challenge in phishing detection is the adversarial nature of the domain. Attackers continuously alter their writing style, vocabulary, and technical evasion tricks (e.g., using typos, embedding HTML elements, inserting invisible characters) to bypass security filters. Furthermore, a high rate of False Positives (classifying a critical business email as spam) disrupts operations, while False Negatives (letting a phishing email land in the inbox) pose critical security risks.

### B. Formal Definition
Let $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$ be a dataset of $N$ emails, where $x_i$ represents the raw text of the $i$-th email and $y_i \in \{0, 1\}$ represents the ground-truth class label, where:
* $y_i = 0$ designates a **Ham** (legitimate) email.
* $y_i = 1$ designates a **Spam** (phishing) email.

The goal is to learn a mapping function $f: X \to Y$ using training data such that the classification error on unseen emails is minimized, and the F1-score is maximized.

### C. Primary Objectives
The primary objectives of this final year project are:
1. **Develop an Automated Preprocessing Pipeline:** Clean raw, noisy text by removing formatting artifacts (HTML, URLs), filtering non-alphabetic elements, and normalizing vocabulary through lemmatization.
2. **Conduct Comparative Feature Engineering:** Implement and contrast text vectorization models to identify the optimal representation.
3. **Compare Machine Learning Architectures:** Evaluate the predictive power of linear models, probabilistic models, tree-based models, and maximum-margin classifiers.
4. **Achieve High Practical Accuracy:** Ensure the finalized model achieves a classification accuracy exceeding **95%** and an F1-score exceeding **90%** on validation sets.
5. **Implement an Interactive UI:** Package the machine learning backend into a user-friendly Streamlit web interface to demonstrate real-world deployment viability.

---

## III. DATASET CHARACTERISTICS & PREPROCESSING PIPELINE

### A. Dataset Selection
The project uses the `dataset/enron_spam_data.csv` file, which is based on the Enron Email Dataset, representing a standard benchmark for email text classification tasks [2].

#### 1) Raw Dataset Characteristics
An initial programmatic inspection of the raw dataset yields the following characteristics:
* **Total Records:** 33,716 rows
* **Total Columns:** 5 columns
* **Column Labels:**
  1. `Message ID`: Unique identifier for each email.
  2. `Subject`: The subject line of the email (289 nulls).
  3. `Message`: The main body content of the email (371 nulls).
  4. `Spam/Ham`: The target classification label (`spam` or `ham`).
  5. `Date`: The timestamp of when the email was sent.

#### 2) Data Cleansing and Label Encoding
During the cleaning phase, we combine the `Subject` and `Message` columns into a single text column by replacing any nulls with empty spaces. This ensures the model learns features from both the subject lines and the email body. Duplicate rows and null content rows are also discarded to prevent data leakage and artificially inflated performance scores. Removing these rows results in a cleaned corpus of **30,494 unique records**.

The textual label in the target column `Spam/Ham` is encoded into a binary format:
$$\text{Label}(\text{Spam/Ham}) = \begin{cases} 0, & \text{if } \text{Spam/Ham} = \text{"ham"} \\ 1, & \text{if } \text{Spam/Ham} = \text{"spam"} \end{cases}$$

#### 3) Class Distribution Analysis
The final dataset is well-balanced, which provides a highly stable training domain for our machine learning classifiers:
* **Ham (Legitimate):** 15,910 emails (52.17%)
* **Spam (Phishing):** 14,584 emails (47.83%)
* **Ratio:** Approximately 1.1:1 (Legitimate to Phishing)

### B. Preprocessing Pipeline Steps
Text preprocessing is a critical step in NLP [3]. Raw text contains a large amount of noise—such as formatting code, grammar particles, punctuation, and structural variations—that adds complexity without contributing to semantic interpretation. The pipeline consists of:

1. **Case Normalization:** Convert all characters to lowercase. This ensures that words like "Urgent", "URGENT", and "urgent" are treated identically.
2. **HTML Tag Removal:** Phishing emails often contain embedded HTML code (e.g., `<a href="...">`, `<br>`, `<strong>`) to spoof visual layouts. We strip these tags using regular expressions: `re.sub(r'<[^>]+>', '', text)`.
3. **URL Stripping:** Hyperlinks are replaced with empty strings. Since actual links change constantly, the raw string of the URL increases vocabulary size unnecessarily. We remove them using: `re.sub(r'https?://\S+|www\.\S+', '', text)`.
4. **Number Removal:** Numerical values are stripped using regex: `re.sub(r'\d+', '', text)`.
5. **Punctuation Filtering:** Punctuation marks (`!`, `@`, `#`, `$`, `%`, etc.) are removed to prevent variations of words (e.g., "winner!" vs "winner") from being indexed as separate features.
6. **Tokenization:** Split sentences into individual words (tokens) using NLTK's `word_tokenize`.
7. **Stopword Elimination:** Eliminate common words that appear frequently across all documents but carry little semantic value (e.g., "the", "is", "and"). We use NLTK's standard English stopword list.
8. **Lemmatization:** Reduce words to their base or dictionary form (lemma) using NLTK's `WordNetLemmatizer`. For example, "running", "ran", and "runs" are mapped to "run".

### C. Malicious URL Classification Dataset
To extend the capability of PhishGuard AI, we integrated a secondary dataset specifically for link-level analysis: `dataset/malicious_phish.csv`. This dataset consists of **651,191 URLs** (428,103 benign, 96,457 defacement, 94,111 phishing, and 32,520 malware URLs). After removing duplicate records, the corpus contains **641,119 unique URLs** mapped to **154,471 unique domains**. This large dataset is used to train a robust URL classifier model that operates alongside text analysis.

---

## IV. FEATURE REPRESENTATIONS & EXTRACTION METHODOLOGY

Machine learning algorithms require numerical input. Text vectorization is the process of mapping textual content into a high-dimensional vector space.

### A. CountVectorizer (Bag of Words)
The Bag-of-Words (BoW) model represents text by tracking word occurrences. It constructs a vocabulary matrix $V$ of all unique terms across the corpus, representing each document $d$ as a vector:
$$\mathbf{x}_d = [f(t_1, d), f(t_2, d), \dots, f(t_{|V|}, d)]$$
where $f(t_j, d)$ is the raw frequency of term $t_j$ in document $d$. 

*   **Limitation:** BoW counts words without considering their relative importance across the entire corpus. Highly frequent words across all documents can dominate the representation, overshadowing rare, highly informative keywords.

### B. TF-IDF Vectorizer
The Term Frequency-Inverse Document Frequency (TF-IDF) representation addresses this limitation by weighting term frequency with its inverse document frequency:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
1. **Term Frequency (TF):** The frequency of term $t$ in document $d$.
   $$\text{TF}(t, d) = \frac{f(t, d)}{\sum_{t' \in d} f(t', d)}$$
2. **Inverse Document Frequency (IDF):** Measures how much information a word provides across the entire corpus $D$.
   $$\text{IDF}(t, D) = \log \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$

Terms that appear in almost all documents receive low IDF weights, while terms that appear frequently within a specific class of documents (e.g., "prize", "verify", "account", "login") receive high weights.

### C. Why TF-IDF Outperforms CountVectorizer
TF-IDF is theoretically and practically superior for email text classification:
1. **Incorporate Document Length Normalization:** TF-IDF divides term frequency by document length, preventing longer emails from dominating classification decisions.
2. **Penalty on Ubiquitous Vocabulary:** It down-weights terms that occur globally across both ham and spam emails (such as "would", "think"), highlighting class-specific words (like "claim", "urgent").
3. **Sparse Representation Stability:** TF-IDF outputs normalized values bounded between $[0, 1]$, which enhances the stability and convergence rate of optimization solvers in models like Logistic Regression and SVM.

---

## V. PROPOSED METHODOLOGY & SYSTEM ARCHITECTURE

The structural design of the PhishGuard AI system is illustrated in the architecture diagram below.

```mermaid
graph TD
    A[Raw Email Input] --> B[Text Preprocessing]
    B --> B1[HTML & URL Stripping]
    B --> B2[Case Normalization]
    B --> B3[Tokenization & Stopword Removal]
    B --> B4[WordNet Lemmatization]
    
    B4 --> C[Feature Extraction]
    C --> C1[Count Vectorization]
    C --> C2[TF-IDF Vectorization]
    
    C1 --> D[Data Split 80/20]
    C2 --> D
    
    D --> E[Model Training & Evaluation]
    E --> E1[Logistic Regression]
    E --> E2[Multinomial Naive Bayes]
    E --> E3[Random Forest]
    E --> E4[Decision Tree]
    E --> E5[Linear SVM]
    
    E1 & E2 & E3 & E4 & E5 --> F[Performance Metric Comparison]
    F --> G[Pick Best Model]
    G --> H[Deployment in Streamlit UI (SVM + BERT Dual-Engine)]
```

### A. Core System Phases
The system operates in two core phases: **Training Phase** and **Prediction Phase**.
1) *Training Phase:* Raw text is preprocessed, converted into numerical vectors, split into training/testing sets, modeled using ML classifiers, and evaluated. The best model and vectorizer are then serialized.
2) *Prediction Phase:* An end-user inputs text into the Streamlit application. The system processes the text simultaneously through two independent detection engines: (a) the traditional TF-IDF + LinearSVC pipeline, and (b) a deep learning transformer (BERT/DistilBERT). The predictions from both models are compared, and a consensus safety verdict (High-Confidence Phishing, Suspicious, or Safe) is determined and displayed alongside side-by-side confidence scores. Concurrently, any hyperlinks detected in the email body are evaluated by a hybrid URL analysis module consisting of: (a) lexical heuristics (e.g., protocol checks, obfuscation service detection, raw IP hosting, subdomain depth), and (b) an XGBoost classification model trained on a 651k URL database to predict URL threat categories (phishing, defacement, malware, or benign). Furthermore, a dedicated Email Header and Metadata Spoofing Auditor is integrated to inspect sender credentials. This auditor checks for Display Name spoofing, domain typosquatting using a homoglyph-aware Levenshtein similarity algorithm, corporate brand domain abuse from free webmail providers (e.g., Gmail, Yahoo), and Reply-To header mismatches where the return path is redirected to a different core domain.

### B. Chrome Extension Client-Server Architecture
To transition PhishGuard AI's analytical capabilities into the user's active browsing workflow, we developed a Google Chrome Extension built on the Manifest V3 specification. 
Because web extensions operate in a sandboxed JavaScript runtime environment, executing heavy Python models (SVM, XGBoost, and BERT) client-side is unviable. We resolved this constraint by designing a client-server architecture:
*   **FastAPI REST Server (`api.py`):** The Python models are wrapped in a high-performance REST API hosted locally using FastAPI and Uvicorn, with Cross-Origin Resource Sharing (CORS) enabled.
*   **Extension Client:** The extension is structured with a content script (`content.js`) that runs in the page DOM to extract text and header attributes from active Gmail or Microsoft Outlook tabs, and a popup window (`popup.html`/`popup.js`) designed with premium glassmorphism styling that queries the backend FastAPI endpoints asynchronously.

### C. Adaptive Continuous Learning & Dynamic Threat Intelligence
To enable PhishGuard AI to adapt to new phishing campaigns and zero-day threat variants dynamically, we integrated a real-time active learning and feedback pipeline:
1) *Dynamic Rule Override Database (`models/local_intel.json`):* When a user flags a misclassification or identifies a new threat, the domain and email signature are logged into a local thread intelligence database. Future scans immediately evaluate links and senders against this blacklist, offering sub-millisecond protection.
2) *User Correction Corpus (`dataset/feedback_data.csv`):* User corrections (e.g., labeling safe mail as phishing or vice versa) are logged to a CSV format feedback file.
3) *Asynchronous Background Retraining:* When feedback is submitted, the API spawns an asynchronous background thread that merges the baseline Enron dataset with the feedback dataset, re-fits the TF-IDF feature extractor (learning new vocabularies), retrains the LinearSVC model weights, serializes new `.pkl` checkpoints, and clears Streamlit cache loaders to load the updated weights hot-swapped in memory.

---

## VI. MACHINE LEARNING CLASSIFIERS

We train and compare five representative supervised learning classifiers:

### A. Logistic Regression
A linear model that models the probability of a binary target variable using the logistic sigmoid function:
$$P(Y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
It is highly efficient, interpretable, and serves as an excellent baseline.

### B. Multinomial Naive Bayes (MNB)
A probabilistic classifier based on Bayes' Theorem, operating under the assumption of strong conditional independence between features:
$$P(Y | X_1, \dots, X_n) \propto P(Y) \prod_{i=1}^n P(X_i | Y)$$
MNB works exceptionally well with text bag-of-words or TF-IDF count features, modeling term frequencies using a multinomial distribution [5].

### C. Random Forest Classifier
An ensemble learning technique that trains multiple decision trees in parallel using bootstrap aggregating (bagging) and feature randomness. The final classification is determined by majority vote:
$$f(\mathbf{x}) = \text{mode}\{T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x})\}$$
It is robust to overfitting, handles high-dimensional spaces well, and provides feature importance scores.

### D. Decision Tree Classifier
A non-parametric model that recursively partitions the feature space based on impurity measures (such as Gini impurity or Information Gain). It is highly interpretable but prone to overfitting on text data due to high dimensionality.

### E. Support Vector Machine (LinearSVC)
A classifier that fits a maximum-margin hyperplane in the feature space to separate classes:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
Linear SVC is particularly suited for text classification because text features are highly dimensional and often linearly separable. The LinearSVC formulation optimizes the dual problem efficiently, leading to high training speeds and state-of-the-art accuracy [4].

### F. Bidirectional Encoder Representations from Transformers (BERT)
BERT is a deep learning transformer model designed to pre-train bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. In this project, a fine-tuned lightweight DistilBERT classifier is integrated during the live prediction phase. This captures deep semantic context, contextual shifts, and suspicious intent that traditional word-matching models cannot capture.

### G. XGBoost URL Classifier
For link-level analysis, we utilize **XGBoost (Extreme Gradient Boosting)**, a scalable tree boosting system. The feature space consists of a hybrid matrix of 13 lexical features (e.g., URL length, subdomain depth, digit counts, raw IP hosting, brand terms checks) and a character-level TF-IDF vectorizer (3-to-5 character n-grams, capped at 3,000 features). To prevent **domain leakage**, training/validation splits are executed on a disjoint domain basis (80% train domains, 20% test domains). This guarantees that domains present in training are never seen in validation, forcing the tree booster to learn generalized structural patterns of malicious links.

---

## VII. EVALUATION METRICS & EXPERIMENTAL RESULTS

To evaluate model performance, we use a standard testing set (20% of the dataset, random state 42). The performance is measured using the following metrics derived from the confusion matrix:

* **True Positive (TP):** Phishing email correctly classified as Spam.
* **False Positive (FP):** Legitimate email incorrectly classified as Spam.
* **True Negative (TN):** Legitimate email correctly classified as Ham.
* **False Negative (FN):** Phishing email incorrectly classified as Ham.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{Measures quality of spam predictions})$$
$$\text{Recall} = \frac{TP}{TP + FN} \quad (\text{Measures spam coverage})$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (\text{Harmonic mean of Precision and Recall})$$

### A. Quantitative Performance Comparison
The tables below present a comprehensive comparison of model performance on the test dataset.

##### Table 1: Model Comparison using TF-IDF Features
| Model Name | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **99.13%** | **98.68%** | **99.52%** | **99.10%** |
| **Logistic Regression** | 98.92% | 98.28% | 99.49% | 98.88% |
| **Multinomial Naive Bayes** | 98.85% | 98.90% | 98.70% | 98.80% |
| **Random Forest** | 98.57% | 98.13% | 98.91% | 98.52% |
| **Decision Tree** | 95.80% | 96.19% | 95.01% | 95.60% |

##### Table 2: Malicious URL Classifier Performance (XGBoost)
| Class Label | Precision | Recall | F1-Score | Validation Support |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 96.00% | 99.00% | 97.00% | 32,955 |
| **Phishing** | 90.00% | 79.00% | 84.00% | 7,219 |
| **Defacement** | 98.00% | 97.00% | 97.00% | 8,483 |
| **Malware** | 98.00% | 86.00% | 91.00% | 1,341 |
| **Overall Accuracy** | | | **95.41%** | 49,998 |

### B. Discussion and Best Model Selection
1. **Performance with TF-IDF:** On the TF-IDF representation, the Linear SVC model outperformed all others with an F1-score of **99.10%** and an accuracy of **99.13%**. It successfully detected 2,910 out of 2,924 spam messages (99.52% Recall) while generating only 26 false positives (98.68% Precision).
2. **Performance across Models:** All models perform exceptionally well on the Enron email dataset, with classifiers like SVM, Logistic Regression, and Naive Bayes exceeding 98% accuracy. This indicates the high quality and linearly separable characteristics of the text features extracted using the combined Subject + Message field.
3. **Generalization (Overfitting Check):** The SVM model was evaluated on both training and testing subsets to detect overfitting. The training accuracy reached **99.99%** while the validation testing accuracy reached **99.13%**. The minor difference of **0.86%** indicates excellent model generalization.
4. **Final Model Choice:** Based on its balanced performance and high F1-score, the **Support Vector Machine (LinearSVC)** trained on TF-IDF features was selected as the production model for deployment.

---

## VIII. CONCLUSION & FUTURE DIRECTIONS

### A. Conclusion
We have successfully developed, analyzed, and deployed an end-to-end NLP framework for phishing email detection. By systematically implementing rigorous data cleansing, tokenization, stopword elimination, and WordNet-based lemmatization, the raw text was converted into highly descriptive features. Linear SVC combined with TF-IDF features proved to be the most robust architecture, yielding **99.13% accuracy** and a balanced **99.10% F1-score**. 

Furthermore, we integrated:
1) An **Email Header and Metadata Spoofing Auditor** executing homoglyph-aware Levenshtein typosquatting checks and Reply-To domain redirect checks to mitigate social engineering and Business Email Compromise (BEC) risks.
2) A **Manifest V3 Chrome Extension** supported by a **FastAPI REST API Server** to enable real-time active DOM email scanning directly inside browser clients.
3) An **Adaptive Continuous Learning Pipeline** that updates local threat intelligence databases (`local_intel.json`) instantly and triggers background model retraining upon user feedback submission.

The integration of these modules provides an enterprise-grade, adaptive ecosystem for email security analysis.

### B. Future Scope
To improve model robustness and expand capabilities in future iterations, we propose:
1. **Contextual Deep Learning Architectures:** Evaluate Recurrent Neural Networks (LSTMs) or Transformer-based models (like RoBERTa) to capture contextual semantics and long-range dependencies in email text.
2. **Standard Protocols Verification:** Integrate validation checks for email authentication standards—namely SPF (Sender Policy Framework), DKIM (DomainKeys Identified Mail), and DMARC (Domain-based Message Authentication, Reporting, and Conformance)—by inspecting raw email headers.
3. **Defense Against Adversarial Attacks:** Train models using adversarial samples (e.g., text with intentional typos or hidden characters) to build resilience against evasion techniques.

---

## REFERENCES

*   [1] D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. Prentice Hall, 2024.
*   [2] V. Metsis, I. Androutsopoulos, and G. Paliouras, "Spam filtering with Naive Bayes - Which Naive Bayes?", *Proceedings of the 3rd Conference on Email and Spam (CEAS)*, 2006, pp. 27-28.
*   [3] S. Bird, E. Klein, and E. Loper, *Natural language processing with Python: analyzing text with the natural language toolkit*. O'Reilly Media, Inc., 2009.
*   [4] C. Cortes and V. Vapnik, "Support-vector networks", *Machine learning*, vol. 20, no. 3, pp. 273-297, 1995.
*   [5] A. McCallum and K. Nigam, "A comparison of event models for naive bayes text classification", *AAAI-98 workshop on learning for text categorization*, vol. 752, pp. 41-48, 1998.
