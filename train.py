import os
import re
import string
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# Natural Language Toolkit
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-Learn Utilities and Models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report)

# Setup NLTK paths
nltk_data_dir = os.path.expanduser("~/nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Ensure NLTK datasets are downloaded/cached
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
except Exception:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    # Lowercase text
    text = text.lower()
    # Remove HTML tags using regular expression
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs using regular expression
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove numbers using regular expression
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and Lemmatize tokens
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(cleaned_tokens)

def main():
    print("Loading Enron dataset...")
    df = pd.read_csv(os.path.join('dataset', 'enron_spam_data.csv'))
    
    # 1. Combine Subject and Message for full context
    df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
    
    # 2. Clean up dataset
    df = df[['text', 'Spam/Ham']].dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 3. Map labels: ham -> 0, spam -> 1
    df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset shape after preprocessing: {df.shape[0]} rows")
    print("Class distribution:\n", df['Spam/Ham'].value_counts())
    
    print("Preprocessing text data (this may take a minute)...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    X = df['cleaned_text']
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Set Size: {X_train.shape[0]} samples")
    print(f"Testing Set Size: {X_test.shape[0]} samples")
    
    # Extract Features using TF-IDF
    print("Vectorizing text with TF-IDF...")
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)
    print(f"Vocabulary size: {len(tfidf_vect.vocabulary_)} features")
    
    # Train SVM Model
    print("Training Support Vector Machine (LinearSVC)...")
    model = LinearSVC(C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n================== Evaluation Results ==================")
    print(f"Accuracy  : {acc:.4%}")
    print(f"Precision : {prec:.4%}")
    print(f"Recall    : {rec:.4%}")
    print(f"F1-Score  : {f1:.4%}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Save the models
    os.makedirs('models', exist_ok=True)
    print("Saving models to directory...")
    with open(os.path.join('models', 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vect, f)
    print("Models saved successfully!")

if __name__ == '__main__':
    main()
