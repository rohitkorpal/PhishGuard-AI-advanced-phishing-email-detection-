import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Natural Language Toolkit
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

# Setup NLTK paths
nltk_data_dir = os.path.expanduser("~/nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(cleaned_tokens)

def main():
    print("Loading dataset...")
    df = pd.read_csv(os.path.join('dataset', 'enron_spam_data.csv'))
    df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')
    df = df[['text', 'Spam/Ham']].dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
    
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Class Distribution Plot
    print("Generating class distribution plot...")
    plt.figure(figsize=(6, 4))
    sns.set_theme(style='whitegrid')
    ax = sns.countplot(x='Spam/Ham', data=df, palette=['#4ea8de', '#560bad'])
    plt.title('Enron Dataset - Class Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Category', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='semibold')
    plt.tight_layout()
    plt.savefig('visualizations/enron_class_distribution.png', dpi=150)
    plt.close()
    
    # 2. Confusion Matrix
    print("Preprocessing text for CM...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)
    
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)
    
    model = LinearSVC(C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Generating confusion matrix plot...")
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix (LinearSVC on Enron)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    plt.savefig('visualizations/enron_confusion_matrix.png', dpi=150)
    plt.close()
    
    # 3. WordClouds
    print("Generating WordClouds (this might take a bit)...")
    spam_text = ' '.join(df[df['label'] == 1]['cleaned_text'].astype(str).tolist()[:5000]) # Limit to 5000 to be faster
    ham_text = ' '.join(df[df['label'] == 0]['cleaned_text'].astype(str).tolist()[:5000])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    wc_spam = WordCloud(width=500, height=350, background_color='black', colormap='Reds', max_words=80).generate(spam_text)
    axes[0].imshow(wc_spam, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Spam (Phishing) WordCloud', fontsize=12, fontweight='bold', pad=10)
    
    wc_ham = WordCloud(width=500, height=350, background_color='black', colormap='GnBu', max_words=80).generate(ham_text)
    axes[1].imshow(wc_ham, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Ham (Legitimate) WordCloud', fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/enron_wordclouds.png', dpi=150)
    plt.close()
    print("All plots generated and saved successfully!")

if __name__ == '__main__':
    main()
