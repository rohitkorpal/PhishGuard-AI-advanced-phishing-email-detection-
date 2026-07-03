import pandas as pd
import numpy as np
import os
import re
import pickle
import time
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import scipy.sparse as sp
import xgboost as xgb

def extract_hostname(url):
    url_clean = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    url_clean = re.sub(r'^www\.', '', url_clean, flags=re.IGNORECASE)
    parts = re.split(r'[:/]', url_clean)
    return parts[0].lower()

def extract_core_domain(url):
    hostname = extract_hostname(url)
    parts = hostname.split('.')
    if len(parts) >= 2:
        tld_components = {'com', 'co', 'org', 'net', 'gov', 'edu', 'ac', 'mil'}
        if len(parts) >= 3 and parts[-2] in tld_components:
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return hostname

def get_lexical_features(url):
    hostname = extract_hostname(url)
    url_len = len(url)
    host_len = len(hostname)
    
    dot_count = url.count('.')
    slash_count = url.count('/')
    hyphen_count = url.count('-')
    underscore_count = url.count('_')
    question_count = url.count('?')
    equal_count = url.count('=')
    amp_count = url.count('&')
    digit_count = sum(c.isdigit() for c in url)
    
    is_ip = 1 if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', hostname) else 0
    
    suspicious_words = ['login', 'secure', 'verify', 'account', 'update', 'signin', 'banking', 'click', 'confirm', 'free', 'gift', 'award', 'prize']
    has_susp_word = 1 if any(word in url.lower() for word in suspicious_words) else 0
    
    subdomain_count = max(0, hostname.count('.') - 1)
    
    return [
        url_len, host_len, dot_count, slash_count, hyphen_count,
        underscore_count, question_count, equal_count, amp_count,
        digit_count, is_ip, has_susp_word, subdomain_count
    ]

def train_model():
    print("Step 1: Loading malicious_phish.csv dataset...")
    df = pd.read_csv(os.path.join("dataset", "malicious_phish.csv"))
    print(f"Total raw dataset size: {len(df)}")
    
    # 1. Duplicate Removal
    print("Step 2: Removing duplicate URL records...")
    df = df.drop_duplicates(subset=['url']).reset_index(drop=True)
    print(f"Dataset size after duplicate removal: {len(df)}")
    
    # 2. Domain Extraction
    print("Step 3: Extracting core domains for split...")
    df['core_domain'] = df['url'].apply(extract_core_domain)
    
    # 3. Domain Split to Prevent Leakage
    print("Step 4: Creating domain-level split...")
    unique_domains = df['core_domain'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_domains)
    
    split_idx = int(len(unique_domains) * 0.8)
    train_domains = set(unique_domains[:split_idx])
    
    df['is_train'] = df['core_domain'].apply(lambda d: d in train_domains)
    train_df = df[df['is_train']].copy()
    test_df = df[~df['is_train']].copy()
    
    print(f"Unique Domains: {len(unique_domains)}")
    print(f"Training Domains: {len(train_domains)} | Testing Domains: {len(unique_domains) - len(train_domains)}")
    print(f"Training URLs: {len(train_df)} | Testing URLs: {len(test_df)}")
    
    # Downsample slightly to optimize memory and speed if dataset is extremely large
    # Using 250,000 URLs (maintaining class ratios) is more than enough for a state-of-the-art classifier
    MAX_SAMPLES = 250000
    if len(train_df) > MAX_SAMPLES:
        print(f"Downsampling training URLs to {MAX_SAMPLES} to optimize memory...")
        train_df = train_df.groupby('type', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(MAX_SAMPLES * (len(x)/len(train_df)))), random_state=42)
        )
        print(f"New training set size: {len(train_df)}")
        
    if len(test_df) > 50000:
        print(f"Downsampling testing URLs to 50,000 to speed up validation...")
        test_df = test_df.groupby('type', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(50000 * (len(x)/len(test_df)))), random_state=42)
        )
        print(f"New testing set size: {len(test_df)}")

    # 4. Feature Extraction
    print("Step 5: Extracting lexical features...")
    X_train_lex = np.array([get_lexical_features(u) for u in train_df['url']])
    X_test_lex = np.array([get_lexical_features(u) for u in test_df['url']])
    
    print("Step 6: Fitting Character N-Gram TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(train_df['url'])
    X_test_tfidf = vectorizer.transform(test_df['url'])
    
    # Combine lexical and TF-IDF features
    print("Step 7: Combining features...")
    X_train = sp.hstack([X_train_lex, X_train_tfidf], format='csr')
    X_test = sp.hstack([X_test_lex, X_test_tfidf], format='csr')
    
    # Map classes
    class_map = {'benign': 0, 'phishing': 1, 'defacement': 2, 'malware': 3}
    y_train = train_df['type'].map(class_map).values
    y_test = test_df['type'].map(class_map).values
    
    # 5. Train XGBoost model
    print("Step 8: Initializing XGBoost Classifier...")
    
    # Enable GPU tree method
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda', # Run on Nvidia GeForce RTX 5050 Laptop GPU!
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    print("Step 9: Training XGBoost on GPU...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"Training completed successfully in {elapsed:.2f} seconds!")
    
    # 6. Evaluation
    print("Step 10: Evaluating classifier performance...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"URL Classifier Validation Accuracy: {acc*100:.2f}%")
    
    inverse_class_map = {v: k for k, v in class_map.items()}
    target_names = [inverse_class_map[i] for i in range(4)]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 7. Serialization
    print("Step 11: Saving model artifacts...")
    os.makedirs("models", exist_ok=True)
    
    with open(os.path.join("models", "url_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
        
    with open(os.path.join("models", "url_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("URL Classifier files successfully saved to 'models/url_classifier.pkl' and 'models/url_vectorizer.pkl'!")

if __name__ == "__main__":
    train_model()
