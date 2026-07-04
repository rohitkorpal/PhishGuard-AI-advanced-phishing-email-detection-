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
import json
import csv
import threading

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

# Try to import fpdf2; auto-install if missing
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    print("PDF library (fpdf2) not found. Installing automatically...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
        from fpdf import FPDF
        HAS_FPDF = True
        print("fpdf2 library installed successfully!")
    except Exception as e:
        print(f"Auto-installation of fpdf2 failed: {e}")
        HAS_FPDF = False

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

from urllib.parse import urlparse
import datetime

def extract_core_domain(url):
    try:
        url_clean = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
        url_clean = re.sub(r'^www\.', '', url_clean, flags=re.IGNORECASE)
        # Strip display names or brackets in case raw emails are passed
        if '<' in url_clean and '>' in url_clean:
            url_clean = url_clean.split('<')[-1].split('>')[0]
        if '@' in url_clean:
            url_clean = url_clean.split('@')[-1]
            
        parts = re.split(r'[:/]', url_clean)
        hostname = parts[0].lower().strip()
        
        host_parts = hostname.split('.')
        if len(host_parts) > 2:
            # Handle common double extensions like co.uk, co.in, com.br
            if host_parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu', 'ac', 'res']:
                return '.'.join(host_parts[-3:])
            return '.'.join(host_parts[-2:])
        return hostname
    except Exception:
        return url

def get_url_lexical_features(url):
    url_clean = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    url_clean = re.sub(r'^www\.', '', url_clean, flags=re.IGNORECASE)
    parts = re.split(r'[:/]', url_clean)
    hostname = parts[0].lower()
    
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

def scan_urls_in_text(text):
    # Find all URLs in the email text
    url_pattern = r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/\S*)?'
    # Clean matches from common text characters like brackets
    raw_urls = re.findall(url_pattern, text)
    
    if not raw_urls:
        return []
        
    audit_results = []
    
    # Official brand domains whitelist (brands to monitor for spoofing attempts)
    official_brands = {
        "paypal": ["paypal.com", "paypal.in", "paypal-status.com"],
        "google": ["google.com", "google.co.in", "gmail.com", "youtube.com"],
        "netflix": ["netflix.com", "netflix.net"],
        "microsoft": ["microsoft.com", "live.com", "outlook.com", "office.com", "sharepoint.com"],
        "amazon": ["amazon.com", "amazon.in", "aws.amazon.com"],
        "apple": ["apple.com", "icloud.com"],
        "facebook": ["facebook.com", "fb.com"],
        "meta": ["meta.com", "instagram.com", "whatsapp.com"],
        "yahoo": ["yahoo.com", "yahoo.co.in"]
    }
    
    # Common URL shorteners to check
    shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"]
    
    seen_urls = set()
    for url in raw_urls:
        # Ignore placeholders like [FAKE-LINK-HERE] or simple brackets
        url_clean = url.strip("[]()\"' ,.;:-")
        if not url_clean or url_clean in seen_urls:
            continue
            
        # Ignore text indicators that are not actual links
        if "." not in url_clean or len(url_clean) < 4:
            continue
            
        seen_urls.add(url_clean)
        
        analysis = {
            "url": url_clean,
            "status": "Safe",  # Safe, Suspicious, Danger
            "issues": []
        }
        
        # Build standard URL for parsing
        parse_url = url_clean
        if not parse_url.startswith(("http://", "https://")):
            parse_url = "http://" + parse_url
            
        try:
            parsed = urlparse(parse_url)
            hostname = parsed.hostname.lower() if parsed.hostname else ""
        except Exception:
            hostname = ""
            
        if not hostname:
            continue
            
        # Check Local Threat Intelligence Database for manual blacklists
        intel_data = load_local_intel()
        blacklisted_domains = [d.lower() for d in intel_data.get("blacklisted_domains", [])]
        core_domain = extract_core_domain(url_clean)
        if (core_domain.lower() in blacklisted_domains) or (hostname and hostname.lower() in blacklisted_domains):
            analysis["issues"].append("🚨 Local Threat Intelligence: Target domain is manually blacklisted by user.")
            analysis["status"] = "Danger"
            
        # 1. Insecure Protocol Check
        if url_clean.startswith("http://"):
            analysis["issues"].append("❌ Insecure Protocol: Uses 'http://' instead of secure 'https://'.")
            analysis["status"] = "Suspicious"
            
        # 2. URL Shortener Check
        if any(shortener in hostname for shortener in shorteners):
            analysis["issues"].append("⚠️ Obfuscated Link: Uses a URL shortener service which hides the destination.")
            analysis["status"] = "Suspicious"
            
        # 3. IP Address Domain Check
        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        if re.match(ip_pattern, hostname):
            analysis["issues"].append("🚨 Severe Risk: Uses a raw IP address as a domain, avoiding registration checks.")
            analysis["status"] = "Danger"
            
        # 4. Brand Spoofing Check (Domain Verification)
        for brand, domains in official_brands.items():
            if brand in hostname:
                # If brand keyword is in domain, verify it matches one of the official whitelisted domains
                is_official = any(off_domain in hostname for off_domain in domains)
                if not is_official:
                    analysis["issues"].append(f"🚨 Brand Spoofing: Contains brand keyword '{brand.capitalize()}' but core domain resolves to '{hostname}' instead of official '{domains[0]}'.")
                    analysis["status"] = "Danger"
                    break
                    
        # 5. Over-long subdomain / phishing pattern checks
        if hostname.count(".") >= 4:
            analysis["issues"].append("⚠️ Domain Structure: Excessive number of subdomains (common in credential harvesting spoofing).")
            if analysis["status"] != "Danger":
                analysis["status"] = "Suspicious"
                
        # 6. ML URL Classification Prediction
        if url_model is not None and url_vectorizer is not None:
            try:
                # Extract lexical features
                lex_feats = np.array([get_url_lexical_features(url_clean)])
                # Vectorize character TF-IDF
                tfidf_feats = url_vectorizer.transform([url_clean])
                # Combine using scipy.sparse.hstack
                import scipy.sparse as sp
                combined_feats = sp.hstack([lex_feats, tfidf_feats], format='csr')
                
                # Predict class probabilities
                probs = url_model.predict_proba(combined_feats)[0]
                pred_class_idx = np.argmax(probs)
                pred_prob = probs[pred_class_idx]
                
                class_labels = ['benign', 'phishing', 'defacement', 'malware']
                pred_label = class_labels[pred_class_idx]
                
                # Update status and log details if malicious prediction confidence is high
                if pred_label != 'benign' and pred_prob > 0.6:
                    analysis["issues"].append(f"🤖 Machine Learning Alert: URL classified as '{pred_label.upper()}' with {pred_prob*100:.1f}% confidence.")
                    analysis["status"] = "Danger"
            except Exception:
                pass
                
        audit_results.append(analysis)
        
    return audit_results

def clean_for_pdf(text):
    if not isinstance(text, str):
        return ""
    # Standard FPDF fonts only support Latin-1 (code points < 256)
    # Map common emojis to readable text brackets
    replacements = {
        "🚨": "[ALERT]",
        "⚠️": "[WARNING]",
        "✅": "[SAFE]",
        "❌": "[FAIL]",
        "💡": "[TIP]",
        "⚙️": "[SETTINGS]"
    }
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
        
    cleaned = ""
    for char in text:
        if ord(char) < 256:
            cleaned += char
        else:
            # Fallback for smart punctuation and unsupported symbols
            if char in ['\u201c', '\u201d']:
                cleaned += '"'
            elif char in ['\u2018', '\u2019']:
                cleaned += "'"
            elif char == '\u2013':
                cleaned += '-'
            else:
                cleaned += "?"
    return cleaned

def generate_pdf_report(email_text, svm_res, bert_res, url_res, header_res=None):
    if not HAS_FPDF:
        return None
        
    pdf = FPDF()
    pdf.add_page()
    
    # Title Block
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 58, 138)  # Deep Blue
    pdf.cell(0, 15, "PhishGuard AI: Forensic Security Audit Report", ln=True, align="C")
    
    # Metadata
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(107, 114, 128)  # Grey
    scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ref_id = f"PG-REP-{datetime.datetime.now().strftime('%y%m%d%H%M')}"
    pdf.cell(0, 5, f"Date Generated: {scan_time} | Case Reference: {ref_id}", ln=True, align="C")
    pdf.ln(5)
    pdf.line(10, 32, 200, 32)
    pdf.ln(10)
    
    # Section 1: Consensus Verdict
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Safety Consensus Verdict", ln=True)
    
    pred_svm = svm_res["prediction"]
    pred_bert = bert_res.get("prediction")
    
    pdf.set_font("Helvetica", "B", 12)
    if pred_bert is not None:
        if pred_svm == 1 and pred_bert == 1:
            pdf.set_text_color(220, 38, 38)  # Red
            pdf.cell(0, 8, "VERDICT: [CRITICAL PHISHING ALERT] (Both engines flagged as SPAM)", ln=True)
        elif pred_svm == 1 or pred_bert == 1:
            pdf.set_text_color(217, 119, 6)  # Amber
            pdf.cell(0, 8, "VERDICT: [WARNING] - SUSPICIOUS EMAIL (Single engine flagged as SPAM)", ln=True)
        else:
            pdf.set_text_color(22, 163, 74)  # Green
            pdf.cell(0, 8, "VERDICT: [SAFE EMAIL] (Both engines verified as Ham/Legitimate)", ln=True)
    else:
        if pred_svm == 1:
            pdf.set_text_color(220, 38, 38)
            pdf.cell(0, 8, "VERDICT: [PHISHING ALERT DETECTED] (SVM flagged as SPAM)", ln=True)
        else:
            pdf.set_text_color(22, 163, 74)
            pdf.cell(0, 8, "VERDICT: [SAFE EMAIL] (SVM verified as Ham/Legitimate)", ln=True)
            
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # Section 1.5: Sender Metadata Header Audit
    if header_res is not None:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "1.5. Sender Metadata Verification Logs", ln=True)
        pdf.set_font("Helvetica", "", 10)
        
        pdf.cell(0, 6, f"- Audit Status: {header_res['status'].upper()}", ln=True)
        if header_res["issues"]:
            for issue in header_res["issues"]:
                pdf.cell(0, 6, f"  * {clean_for_pdf(issue)}", ln=True)
        else:
            pdf.cell(0, 6, "  * No structural sender domain discrepancies identified.", ln=True)
        pdf.ln(5)
        
    # Section 2: Engine Performance
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Classifier Confidence Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    svm_lbl = "SPAM / PHISHING" if pred_svm == 1 else "LEGITIMATE (HAM)"
    pdf.cell(0, 6, f"- Support Vector Machine (LinearSVC): {svm_lbl} (Confidence: {svm_res['prob']*100:.2f}%)", ln=True)
    
    if pred_bert is not None:
        bert_lbl = "SPAM / PHISHING" if pred_bert == 1 else "LEGITIMATE (HAM)"
        pdf.cell(0, 6, f"- BERT Contextual Deep Learning: {bert_lbl} (Confidence: {bert_res['prob']*100:.2f}%)", ln=True)
    else:
        pdf.cell(0, 6, "- BERT Contextual Deep Learning: Disabled / Locked (Libraries missing)", ln=True)
        
    pdf.ln(5)
    
    # Section 3: URL Audit Report
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Hyperlink Security Audit Details", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    if not url_res:
        pdf.cell(0, 6, "- No links or web destination addresses were detected inside the email body.", ln=True)
    else:
        for idx, audit in enumerate(url_res, 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"Link {idx}: {clean_for_pdf(audit['url'])}", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 5, f"  Status Assessment: {audit['status'].upper()}", ln=True)
            if audit["issues"]:
                for issue in audit["issues"]:
                    pdf.cell(0, 5, f"  {clean_for_pdf(issue)}", ln=True)
            else:
                pdf.cell(0, 5, "  * No structural anomalies or spoofing flags detected for this domain.", ln=True)
            pdf.ln(2)
            
    pdf.ln(5)
    
    # Section 4: Email Content Preview
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Scanned Email Text Preview", ln=True)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(55, 65, 81)
    
    preview_txt = email_text[:700] + ("\n... [Truncated for report length]" if len(email_text) > 700 else "")
    pdf.multi_cell(0, 5, clean_for_pdf(preview_txt))
    
    # Return raw PDF bytes
    return bytes(pdf.output())

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def clean_homoglyphs(text):
    homoglyphs = {
        '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
        '8': 'b', '9': 'g', 'rn': 'm', 'vv': 'w', 'cl': 'd'
    }
    text_clean = text.lower()
    for hk, hv in homoglyphs.items():
        text_clean = text_clean.replace(hk, hv)
    return text_clean

def audit_email_headers(display_name, email_address, reply_to_address=None):
    if not email_address.strip():
        return None
        
    display_name = display_name.strip()
    email_address = email_address.strip()
    
    if "@" not in email_address:
        return {
            "status": "Danger",
            "issues": ["🚨 Invalid Email Address: Sender address format is invalid (missing '@')."]
        }
        
    parts = email_address.split("@")
    if len(parts) != 2:
        return {
            "status": "Danger",
            "issues": ["🚨 Invalid Email Address: Sender address format is invalid."]
        }
        
    username, domain = parts[0], parts[1].lower()
    
    # Official brand domains whitelist
    official_brands = {
        "paypal": "paypal.com",
        "google": "google.com",
        "gmail": "google.com",
        "netflix": "netflix.com",
        "microsoft": "microsoft.com",
        "outlook": "microsoft.com",
        "amazon": "amazon.com",
        "apple": "apple.com",
        "facebook": "facebook.com",
        "meta": "meta.com",
        "whatsapp": "whatsapp.com",
        "instagram": "instagram.com",
        "yahoo": "yahoo.com",
        "linkedin": "linkedin.com",
        "twitter": "twitter.com"
    }
    
    # Free webmail providers
    free_webmails = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com", "mail.com"}
    
    # High-risk administrative keywords
    executive_keywords = ["ceo", "cfo", "president", "hr", "human resources", "billing", "finance", "security", "support", "admin", "system", "verify", "verification", "login", "alert", "service"]
    
    audit_results = {
        "status": "Safe", # Safe, Suspicious, Danger
        "issues": []
    }
    
    # Check Local Blacklists first
    intel_data = load_local_intel()
    sender_lower = email_address.strip().lower()
    core_domain = extract_core_domain(email_address)
    
    local_blacklist = False
    if sender_lower in [e.lower() for e in intel_data.get("blacklisted_emails", [])]:
        local_blacklist = True
        audit_results["issues"].append(f"🚨 Local Threat Intelligence: Sender email address '{email_address}' is manually blacklisted.")
    if core_domain.lower() in [d.lower() for d in intel_data.get("blacklisted_domains", [])]:
        local_blacklist = True
        audit_results["issues"].append(f"🚨 Local Threat Intelligence: Sender domain '{core_domain}' is manually blacklisted.")
        
    if local_blacklist:
        audit_results["status"] = "Danger"
        
    # Extract Second-Level Domain name (excluding extension like .com)
    domain_sld = core_domain.split('.')[0] if '.' in core_domain else core_domain
    
    # Check 1: Brand Spoofing via Fuzzy Matching & Homoglyphs (Typosquatting)
    sld_normalized = clean_homoglyphs(domain_sld)
    for brand, off_domain in official_brands.items():
        brand_sld = off_domain.split('.')[0]
        brand_normalized = clean_homoglyphs(brand_sld)
        
        # Calculate Levenshtein similarity on normalized SLD segments (split by hyphen)
        parts = sld_normalized.split('-')
        max_sim = 0.0
        for part in parts:
            dist = levenshtein_distance(brand_normalized, part)
            sim = 1.0 - (dist / max(len(brand_normalized), len(part))) if max(len(brand_normalized), len(part)) > 0 else 0
            if sim > max_sim:
                max_sim = sim
                
        if max_sim >= 0.85:
            if core_domain != off_domain and not core_domain.endswith("." + off_domain):
                audit_results["issues"].append(f"🚨 Typosquatting / Brand Impersonation: Domain '{core_domain}' closely mimics official brand '{brand.capitalize()}' (Similarity: {max_sim*100:.1f}% after visual homoglyph correction).")
                audit_results["status"] = "Danger"
                break

    # Check 2: Display Name Spoofing
    display_lower = display_name.lower()
    for brand, off_domain in official_brands.items():
        if brand in display_lower:
            if core_domain != off_domain and not core_domain.endswith("." + off_domain):
                if not any(brand in issue for issue in audit_results["issues"]):
                    audit_results["issues"].append(f"🚨 Display Name Spoofing: Display name claims to represent '{brand.capitalize()}' but actual sending domain is '{core_domain}' (expected: {off_domain}).")
                    audit_results["status"] = "Danger"
                    break
                    
    # Check 3: Free Webmail Abuse for Corporate Claims
    is_free_mail = core_domain in free_webmails
    if is_free_mail:
        has_exec_keyword = any(keyword in display_lower for keyword in executive_keywords)
        has_brand_keyword = any(brand in display_lower for brand in official_brands)
        if has_exec_keyword or has_brand_keyword:
            audit_results["issues"].append(f"🚨 Free Webmail Corporate Abuse: Official corporate or brand claims originating from a public email provider ({domain}). Real services use custom enterprise domains.")
            audit_results["status"] = "Danger"
            
    # Check 4: General Executive Username flags
    username_lower = username.lower()
    if not is_free_mail and audit_results["status"] == "Safe":
        for keyword in ["ceo", "cfo", "president"]:
            if keyword == username_lower or username_lower.startswith(keyword + "-") or username_lower.startswith(keyword + "_"):
                audit_results["issues"].append(f"⚠️ Executive Identity Flag: Sender username starts with high-priority role '{keyword.upper()}'. Validate via alternative contact channel.")
                audit_results["status"] = "Suspicious"
                
    # Check 5: Reply-To Inconsistency Mismatch Check
    if reply_to_address and reply_to_address.strip():
        reply_to_clean = reply_to_address.strip()
        if "@" in reply_to_clean:
            reply_parts = reply_to_clean.split("@")
            if len(reply_parts) == 2:
                reply_core_domain = extract_core_domain(reply_to_clean)
                if reply_core_domain != core_domain:
                    audit_results["issues"].append(f"🚨 Reply-To Header Mismatch: Reply-To address '{reply_to_clean}' points to a different core domain than the From address '{email_address}'. Replies will be routed to an external domain.")
                    audit_results["status"] = "Danger"
                
    return audit_results

# Local Threat Intelligence and Continuous Learning Helper Functions
INTEL_PATH = os.path.join("models", "local_intel.json")
FEEDBACK_CSV_PATH = os.path.join("dataset", "feedback_data.csv")

def load_local_intel():
    if os.path.exists(INTEL_PATH):
        try:
            with open(INTEL_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "blacklisted_domains": [],
        "blacklisted_emails": [],
        "whitelisted_domains": []
    }

def save_local_intel(intel_data):
    try:
        os.makedirs(os.path.dirname(INTEL_PATH), exist_ok=True)
        with open(INTEL_PATH, "w") as f:
            json.dump(intel_data, f, indent=2)
    except Exception as e:
        print(f"Error saving local intel database: {e}")

def submit_local_feedback(text, sender_name, sender_email, reply_to, user_label):
    text = text.strip()
    sender_name = sender_name or ""
    sender_email = sender_email or ""
    reply_to = reply_to or ""
    
    # 1. Append correction to feedback CSV
    file_exists = os.path.exists(FEEDBACK_CSV_PATH)
    try:
        with open(FEEDBACK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(FEEDBACK_CSV_PATH) == 0:
                writer.writerow(["text", "sender_name", "sender_email", "reply_to", "label"])
            writer.writerow([text, sender_name, sender_email, reply_to, user_label])
    except Exception as e:
        print(f"Error appending feedback CSV: {e}")
        
    # 2. Update local_intel.json immediately if labeled as phishing
    intel_updated = False
    if user_label == 1:
        intel_data = load_local_intel()
        
        # Add email
        if sender_email.strip():
            email_clean = sender_email.strip().lower()
            if email_clean not in [e.lower() for e in intel_data["blacklisted_emails"]]:
                intel_data["blacklisted_emails"].append(email_clean)
                intel_updated = True
                
            # Add domain
            domain_core = extract_core_domain(sender_email)
            if domain_core and domain_core not in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "icloud.com"]:
                if domain_core not in [d.lower() for d in intel_data["blacklisted_domains"]]:
                    intel_data["blacklisted_domains"].append(domain_core)
                    intel_updated = True
                    
        # Extract link domains in text and blacklist them
        url_pattern = r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/\S*)?'
        raw_urls = re.findall(url_pattern, text)
        for url in raw_urls:
            url_clean = url.strip("[]()\"' ,.;:-")
            if "." in url_clean and len(url_clean) > 4:
                domain_core = extract_core_domain(url_clean)
                if domain_core and domain_core not in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "icloud.com"]:
                    if domain_core not in [d.lower() for d in intel_data["blacklisted_domains"]]:
                        intel_data["blacklisted_domains"].append(domain_core)
                        intel_updated = True
                        
        if intel_updated:
            save_local_intel(intel_data)
            
    # 3. Trigger asynchronous background model retraining
    t = threading.Thread(target=retrain_models_task)
    t.start()

def retrain_models_task():
    print("Background Job (Streamlit): Initiating PhishGuard AI retraining...")
    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        
        baseline_path = os.path.join("dataset", "enron_spam_data.csv")
        if not os.path.exists(baseline_path):
            print("Retraining aborting: Base Enron file dataset/enron_spam_data.csv is missing.")
            return
            
        df_base = pd.read_csv(baseline_path)
        df_base['text'] = df_base['Subject'].fillna('') + ' ' + df_base['Message'].fillna('')
        df_base = df_base[['text', 'Spam/Ham']].dropna()
        df_base['label'] = df_base['Spam/Ham'].map({'ham': 0, 'spam': 1})
        df_base = df_base[['text', 'label']]
        
        df_feedback = None
        if os.path.exists(FEEDBACK_CSV_PATH) and os.path.getsize(FEEDBACK_CSV_PATH) > 0:
            try:
                df_feedback = pd.read_csv(FEEDBACK_CSV_PATH)
                df_feedback = df_feedback[['text', 'label']].dropna()
                df_feedback['label'] = df_feedback['label'].astype(int)
            except Exception as e:
                print(f"Feedback CSV reading advisory: {e}")
                
        if df_feedback is not None and len(df_feedback) > 0:
            df_combined = pd.concat([df_base, df_feedback], ignore_index=True)
        else:
            df_combined = df_base
            
        df_combined = df_combined.drop_duplicates().reset_index(drop=True)
        df_combined['cleaned_text'] = df_combined['text'].apply(preprocess_text)
        
        X = df_combined['cleaned_text']
        y = df_combined['label']
        
        new_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        X_tfidf = new_vectorizer.fit_transform(X)
        
        new_model = LinearSVC(C=1.0, random_state=42)
        new_model.fit(X_tfidf, y)
        
        model_path = os.path.join("models", "best_model.pkl")
        vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(new_model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(new_vectorizer, f)
            
        # Clear Streamlit's cache
        load_models.clear()
        print("Background Job (Streamlit): Model retraining complete, cache cleared!")
    except Exception as e:
        print(f"Background Job Error: Streamlit retraining failed: {e}")

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
def load_url_models():
    clf_path = os.path.join("models", "url_classifier.pkl")
    vect_path = os.path.join("models", "url_vectorizer.pkl")
    
    if os.path.exists(clf_path) and os.path.exists(vect_path):
        try:
            with open(clf_path, "rb") as f:
                clf = pickle.load(f)
            with open(vect_path, "rb") as f:
                vect = pickle.load(f)
            return clf, vect
        except Exception:
            return None, None
    return None, None

url_model, url_vectorizer = load_url_models()

@st.cache_resource
def load_bert_model():
    if HAS_BERT:
        try:
            device = 0 if torch.cuda.is_available() else -1
            return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection", device=device)
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

# Sidebar Admin Controls (Adaptive Continuous Learning)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Administration")

# Feedback counts
feedback_count = 0
feedback_path = os.path.join("dataset", "feedback_data.csv")
if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            feedback_count = max(0, sum(1 for line in f) - 1)
    except Exception:
        pass

st.sidebar.write(f"📁 Feedback Samples: `{feedback_count}`")

# Blacklisted domains list
intel_data = load_local_intel()
blacklisted_domains = intel_data.get("blacklisted_domains", [])
st.sidebar.write(f"🚫 Blacklisted Domains: `{len(blacklisted_domains)}`")

if st.sidebar.button("🔄 Force Retrain Models", key="sidebar_retrain_btn"):
    with st.spinner("Retraining model in background..."):
        import threading
        t = threading.Thread(target=retrain_models_task)
        t.start()
        st.sidebar.success("Retraining started asynchronously!")

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
            
    st.markdown("---")
    st.header("📊 Malicious URL Dataset Insights")
    st.markdown("""
    To support link-level security checks, the app is integrated with a **651,191 URLs dataset** (`malicious_phish.csv`) that categorizes web addresses into safe and malicious classes.
    """)
    
    # URL Dataset statistics
    col_url1, col_url2, col_url3, col_url4 = st.columns(4)
    with col_url1:
        st.metric("Total Clean Unique URLs", "641,119")
    with col_url2:
        st.metric("Safe / Benign Links", "428,103 (66.8%)")
    with col_url3:
        st.metric("Malicious Links (Phish/Deface/Malware)", "213,016 (33.2%)")
    with col_url4:
        st.metric("Unique Domain Names", "154,471")
        
    st.markdown("""
    *   **Benign URLs (428,103)**: Legitimate standard search engines, corporate sites, and benign references.
    *   **Defacement URLs (96,457)**: Hacked web pages where the visual layout has been compromised or vandalized.
    *   **Phishing URLs (94,111)**: Deceptive landing sites built to harvest credentials (e.g. fake logins for banks or tech companies).
    *   **Malware URLs (32,520)**: Sites hosting malicious payloads, viruses, or ransomware download links.
    """)

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

    st.markdown("---")
    st.header("🔗 Malicious URL Classifier Performance")
    st.markdown("""
    Below are the performance metrics and classification statistics for the **XGBoost URL Classifier** trained on the disjoint domain split of the malicious URL dataset.
    """)
    
    # Callout box for URL model
    st.markdown("""
    <div style='background-color: #f3f0ff; padding: 1.25rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 5px solid #7209b7;'>
        <h4 style='color: #4c1d95; margin: 0; font-family: sans-serif;'>🏆 Selected Classifier: XGBoost (Extreme Gradient Boosting)</h4>
        <p style='color: #4b5563; font-size: 0.95rem; margin-top: 0.5rem; line-height: 1.4; font-family: sans-serif;'>
            XGBoost operates as an optimized distributed gradient boosting library. By combining lexical parser features (URL length, subdomain nested levels, specific characters ratios, raw IP detection, and keywords alerts) with a character 3-to-5-gram TF-IDF vectorizer (capped at 3,000 features), it achieves highly accurate, sub-millisecond class predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        st.metric("Validation Accuracy", "95.41%")
    with col_u2:
        st.metric("Macro F1-Score", "93.00%")
    with col_u3:
        st.metric("Unique Train Domains", "123,576")
        
    st.markdown("### 📊 Classification Metrics Report (By Class)")
    st.markdown("""
    | Class Label | Precision | Recall | F1-Score | Validation Support |
    | :--- | :---: | :---: | :---: | :---: |
    | **Benign (Safe)** | 96.00% | 99.00% | 97.00% | 32,955 |
    | **Phishing** | 90.00% | 79.00% | 84.00% | 7,219 |
    | **Defacement** | 98.00% | 97.00% | 97.00% | 8,483 |
    | **Malware** | 98.00% | 86.00% | 91.00% | 1,341 |
    | **Overall Accuracy** | | | **95.41%** | 49,998 |
    """)
    
    with st.expander("❓ Why did we split train/test based on Unique Domains?"):
        st.markdown("""
        1. **Domain Leakage Avoidance**: Randomly splitting URLs causes links from the same domain to be split across training and validation subsets. Since the model easily memorizes domains, validation scores are artificially inflated (~99%).
        2. **Real-World Generalization**: Splitting the 154,471 unique domains strictly 80/20 guarantees that domains present in training are never seen in testing. This forces the tree booster to learn generalized structural and lexical features of malicious links, rather than memorizing domain strings.
        """)

with tab3:
    st.header("🛡️ Verify Email Content")
    st.markdown("Enter the content of the email you wish to analyze in the text box below. The machine learning model will classify it and output the probability score.")
    
    if model is None or vectorizer is None:
        st.error("Error: Trained model files not found! Please run the training script to generate 'best_model.pkl' and 'tfidf_vectorizer.pkl' in the 'models' directory.")
    else:
        # Text input
        email_text = st.text_area("Email Content", height=200, placeholder="Paste your email text here...")

        # Collapsible Expander for Sender Header inputs
        with st.expander("📧 Advanced Email Header Audit (Optional)"):
            st.markdown("Use these fields to inspect sender metadata (e.g., Display Name vs. Email Domain mismatch checks).")
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                sender_name = st.text_input("Sender Display Name", placeholder="e.g., PayPal Security Alert")
            with col_h2:
                sender_email = st.text_input("Sender Email Address", placeholder="e.g., alert@paypal.com")
            with col_h3:
                reply_to_email = st.text_input("Reply-To Email Address", placeholder="e.g., help@secure-billing.xyz")

        if st.button("Predict / Analyze"):
            if email_text.strip() == "":
                st.warning("Please enter some text to classify.")
            else:
                # -----------------------------------------------------
                # PART 0: Run Sender Metadata Header Check
                # -----------------------------------------------------
                header_audit = None
                if sender_email.strip():
                    header_audit = audit_email_headers(sender_name, sender_email, reply_to_email)
                
                # Check for Local Blacklist matches
                intel_data = load_local_intel()
                sender_domain = extract_core_domain(sender_email) if sender_email else ""
                local_blacklist_triggered = False
                blacklist_reasons = []
                
                if sender_email.strip():
                    email_lower = sender_email.strip().lower()
                    if email_lower in [e.lower() for e in intel_data.get("blacklisted_emails", [])]:
                        local_blacklist_triggered = True
                        blacklist_reasons.append(f"🚨 Local Threat Intelligence: Sender email address '{sender_email}' is manually blacklisted.")
                    if sender_domain.strip():
                        domain_lower = sender_domain.strip().lower()
                        if domain_lower in [d.lower() for d in intel_data.get("blacklisted_domains", [])]:
                            local_blacklist_triggered = True
                            blacklist_reasons.append(f"🚨 Local Threat Intelligence: Sender domain '{sender_domain}' is manually blacklisted.")
                
                if local_blacklist_triggered:
                    if header_audit is None:
                        header_audit = {"status": "Danger", "issues": []}
                    header_audit["status"] = "Danger"
                    for r in blacklist_reasons:
                        if r not in header_audit["issues"]:
                            header_audit["issues"].append(r)
                
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
                # PART 2.5: Run Hyperlink Security Audit (Moved up for verdict)
                # -----------------------------------------------------
                with st.spinner("Analyzing email hyperlinks..."):
                    url_audit_results = scan_urls_in_text(email_text)

                # -----------------------------------------------------
                # PART 3: Consensus Verdict Banner
                # -----------------------------------------------------
                st.markdown("### 🛡️ Unified Safety Verdict")
                
                # Check for threats across all inputs
                has_phishing_verdict = False
                has_suspicious_verdict = False
                
                if local_blacklist_triggered:
                    has_phishing_verdict = True
                elif HAS_BERT and bert_pipeline is not None:
                    if pred_svm == 1 and pred_bert == 1:
                        has_phishing_verdict = True
                    elif pred_svm == 1 or pred_bert == 1:
                        has_suspicious_verdict = True
                else:
                    if pred_svm == 1:
                        has_phishing_verdict = True
                        
                # Check headers status
                if header_audit and header_audit["status"] == "Danger":
                    has_phishing_verdict = True
                elif header_audit and header_audit["status"] == "Suspicious":
                    has_suspicious_verdict = True
                    
                # Check URLs status
                if url_audit_results:
                    if any(audit["status"] == "Danger" for audit in url_audit_results):
                        has_phishing_verdict = True
                    elif any(audit["status"] == "Suspicious" for audit in url_audit_results):
                        has_suspicious_verdict = True

                # Render verdict banner
                if has_phishing_verdict:
                    if local_blacklist_triggered:
                        st.error("🚨 **CRITICAL VERDICT: PHISHING ALERT (Local Threat Intelligence Blacklist Match)**")
                    else:
                        st.error("🚨 **CRITICAL VERDICT: HIGH CONFIDENCE PHISHING ALERT** (ML models, headers, or links flagged phishing)")
                elif has_suspicious_verdict:
                    st.warning("⚠️ **WARNING VERDICT: SUSPICIOUS ACTIVITY** (Model/heuristic mismatch detected. Caution advised)")
                else:
                    st.success("✅ **SAFE VERDICT: LEGITIMATE EMAIL** (Both models and structural checks verified safe)")

                # -----------------------------------------------------
                # PART 3.5: Sender Metadata Header Check Display
                # -----------------------------------------------------
                if header_audit is not None:
                    st.markdown("---")
                    st.markdown("### 📧 Sender Metadata Verification Check")
                    if header_audit["status"] == "Danger":
                        st.error("🚨 **METADATA SPOOFING WARNING** (High-risk vulnerability flags identified on sender address)")
                    elif header_audit["status"] == "Suspicious":
                        st.warning("⚠️ **SENDER INTEGRITY ADVISORY** (Minor domain identity inconsistencies detected)")
                    else:
                        st.success("✅ **SENDER DOMAIN VERIFIED** (Official match for display claims)")
                        
                    for issue in header_audit["issues"]:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;{issue}")

                # -----------------------------------------------------
                # PART 4: Side-by-Side Analysis Panels
                # -----------------------------------------------------
                col_l, col_r = st.columns(2)
                
                with col_l:
                    st.markdown("#### 📊 Traditional ML (LinearSVC)")
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
                
                # -----------------------------------------------------
                # PART 5: Dynamic URL Security Scanner Audit
                # -----------------------------------------------------
                st.markdown("---")
                st.markdown("### 🔗 Hyperlink Security Audit Report")
                
                if not url_audit_results:
                    st.success("✅ **No hyperlinks detected** in the email body. (Low risk of direct credential harvesting/redirects)")
                else:
                    st.info(f"Detected **{len(url_audit_results)}** link(s) in the email content. Review the security audits below:")
                    
                    for idx, audit in enumerate(url_audit_results, 1):
                        st.markdown(f"**Link #{idx}:** `{audit['url']}`")
                        
                        if audit["status"] == "Danger":
                            st.markdown(f"<div style='background-color:#fee2e2; border-left: 5px solid #dc2626; padding:0.75rem; border-radius:4px; color:#991b1b; font-size:0.9rem; font-weight:bold; margin-bottom:0.5rem;'>🚨 DANGER - MALICIOUS LINK PATTERN DETECTED</div>", unsafe_allow_html=True)
                        elif audit["status"] == "Suspicious":
                            st.markdown(f"<div style='background-color:#fef3c7; border-left: 5px solid #d97706; padding:0.75rem; border-radius:4px; color:#92400e; font-size:0.9rem; font-weight:bold; margin-bottom:0.5rem;'>⚠️ SUSPICIOUS - POTENTIAL THREAT VECTOR</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color:#dcfce7; border-left: 5px solid #16a34a; padding:0.75rem; border-radius:4px; color:#166534; font-size:0.9rem; font-weight:bold; margin-bottom:0.5rem;'>✅ VERIFIED SAFE DOMAIN STRUCTURE</div>", unsafe_allow_html=True)
                            
                        if audit["issues"]:
                            for issue in audit["issues"]:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;{issue}")
                        else:
                            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;No brand spoofing, insecure protocols, or obfuscation indicators identified.")
                        st.markdown("<br>", unsafe_allow_html=True)
                
                # -----------------------------------------------------
                # PART 5.5: Continuous Learning Feedback Loop
                # -----------------------------------------------------
                st.markdown("---")
                st.markdown("### 🔄 Help PhishGuard AI Learn")
                st.markdown("Is the classification verdict incorrect? Submit feedback to retrain the machine learning model and update threat intelligence database instantly.")
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    if st.button("🚨 Flag as Phishing (Spam)", key="feedback_phish_btn", use_container_width=True):
                        submit_local_feedback(email_text, sender_name, sender_email, reply_to_email, 1)
                        st.success("Feedback registered! Sender and link domains blacklisted instantly. Model retraining started in the background.")
                with col_f2:
                    if st.button("✅ Mark as Safe (Ham)", key="feedback_safe_btn", use_container_width=True):
                        submit_local_feedback(email_text, sender_name, sender_email, reply_to_email, 0)
                        st.success("Feedback registered! Model retraining started in the background.")
                        
                # -----------------------------------------------------
                # PART 6: Forensic PDF Report Download
                # -----------------------------------------------------
                if HAS_FPDF:
                    st.markdown("---")
                    st.markdown("### 📄 Export Forensic Audit Report")
                    
                    svm_results = {
                        "prediction": int(pred_svm),
                        "prob": float(prob_svm)
                    }
                    
                    bert_results = {}
                    if HAS_BERT and bert_pipeline is not None:
                        bert_results = {
                            "prediction": int(pred_bert),
                            "prob": float(prob_bert)
                        }
                        
                    try:
                        pdf_data = generate_pdf_report(email_text, svm_results, bert_results, url_audit_results, header_audit)
                        if pdf_data:
                            st.download_button(
                                label="Download Forensic PDF Report",
                                data=pdf_data,
                                file_name=f"PhishGuard_Report_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error generating PDF Report: {e}")

                        
st.markdown("---")