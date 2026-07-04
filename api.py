import os
import re
import string
import pickle
import json
import csv
import numpy as np
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GOOGLE_SAFE_BROWSING_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

# NLTK Setup
nltk_data_dir = os.path.expanduser("~/nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Initialize Lemmatizer and Stopwords
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

# Check for deep learning BERT support and GPU usage
try:
    import transformers
    import torch
    from transformers import pipeline
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

# FastAPI Setup
app = FastAPI(
    title="PhishGuard AI API",
    description="Backend API for PhishGuard AI Phishing Email and Link Analyzer with Adaptive Feedback",
    version="1.1.0"
)

# Enable CORS for Chrome Extension requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local Chrome Extension developer mode
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for data and models
INTEL_PATH = os.path.join("models", "local_intel.json")
FEEDBACK_CSV_PATH = os.path.join("dataset", "feedback_data.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

# Local Intel DB Helpers
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

# Load Models
def load_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

model, vectorizer = load_models()

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

def load_bert_model():
    if HAS_BERT:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
            print(f"Loading BERT model on {'GPU (CUDA)' if device == 0 else 'CPU'}...")
            return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection", device=device)
        except Exception as e:
            print(f"BERT pipeline load error: {e}")
            return None
    return None

bert_pipeline = load_bert_model()

# Helper Processing Functions (Mirrors app.py exactly)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Anti-Evasion: Strip zero-width & invisible characters
    invisible_chars = ['\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\ufeff']
    for char in invisible_chars:
        text = text.replace(char, '')
        
    # 2. Anti-Evasion: Unicode Normalization to resolve Homoglyphs (confusables)
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(cleaned_tokens)

def extract_core_domain(url):
    try:
        url_clean = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
        url_clean = re.sub(r'^www\.', '', url_clean, flags=re.IGNORECASE)
        if '<' in url_clean and '>' in url_clean:
            url_clean = url_clean.split('<')[-1].split('>')[0]
        if '@' in url_clean:
            url_clean = url_clean.split('@')[-1]
            
        parts = re.split(r'[:/]', url_clean)
        hostname = parts[0].lower().strip()
        
        host_parts = hostname.split('.')
        if len(host_parts) > 2:
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

def is_gibberish_domain(sld):
    if len(sld) < 10:
        return False
    vowels = set("aeiou")
    vowels_count = sum(1 for c in sld if c in vowels)
    vowel_ratio = vowels_count / len(sld)
    
    consonants = set("bcdfghjklmnpqrstvwxyz")
    max_con = 0
    curr_con = 0
    for c in sld:
        if c in consonants:
            curr_con += 1
            if curr_con > max_con:
                max_con = curr_con
        else:
            curr_con = 0
            
    if vowel_ratio < 0.15:
        return True
    if max_con >= 7:
        return True
    return False

def check_sender_name_address_match(display_name: str, email_address: str) -> bool:
    display_name = display_name.strip().lower()
    email_address = email_address.strip().lower()
    
    if not display_name or "@" not in email_address:
        return True
        
    parts = email_address.split("@")
    username, domain = parts[0], parts[1]
    
    # Exclude common free webmails where personal names are often disjoint from the address
    free_webmails = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com", "mail.com"}
    if domain in free_webmails:
        return True
        
    # Split display name into alphabetical words of length >= 3
    words = re.findall(r'[a-zA-Z]{3,}', display_name)
    if not words:
        return True
        
    # Check if at least one word from display name matches username or domain
    for word in words:
        if word in username or word in domain:
            return True
            
    return False

def check_email_auth_records(domain: str):
    if not domain or domain in {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "icloud.com", "mail.com"}:
        return {"spf": "Valid", "dmarc": "Valid", "warnings": []}
        
    warnings = []
    spf_status = "Missing"
    dmarc_status = "Missing"
    
    import requests
    # 1. Query SPF
    try:
        url = f"https://cloudflare-dns.com/dns-query?name={domain}&type=TXT"
        headers = {"Accept": "application/dns-json"}
        r = requests.get(url, headers=headers, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            answers = data.get("Answer", [])
            for ans in answers:
                txt_data = ans.get("data", "")
                if "v=spf1" in txt_data:
                    spf_status = "Valid"
                    if "+all" in txt_data or "?all" in txt_data:
                        warnings.append(f"⚠️ Weak SPF Record: Domain '{domain}' has a permissive policy ('+all' or '?all') allowing spoofed relays.")
                    break
        if spf_status == "Missing":
            warnings.append(f"🚨 Missing SPF Record: Domain '{domain}' lacks a Sender Policy Framework TXT record, making it easy to spoof.")
    except Exception:
        pass
        
    # 2. Query DMARC
    try:
        url = f"https://cloudflare-dns.com/dns-query?name=_dmarc.{domain}&type=TXT"
        headers = {"Accept": "application/dns-json"}
        r = requests.get(url, headers=headers, timeout=2.5)
        if r.status_code == 200:
            data = r.json()
            answers = data.get("Answer", [])
            for ans in answers:
                txt_data = ans.get("data", "")
                if "v=DMARC1" in txt_data:
                    dmarc_status = "Valid"
                    if "p=none" in txt_data:
                        warnings.append(f"⚠️ Weak DMARC Policy: Domain '{domain}' has 'p=none' which only monitors but doesn't block spoofed emails.")
                    break
        if dmarc_status == "Missing":
            warnings.append(f"🚨 Missing DMARC Policy: Domain '{domain}' has no DMARC record to instruct mail servers to reject spoofed emails.")
    except Exception:
        pass
        
    return {
        "spf": spf_status,
        "dmarc": dmarc_status,
        "warnings": warnings
    }

def audit_email_headers(display_name: str, email_address: str, reply_to_address: Optional[str] = None, mailed_by: Optional[str] = None):
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
    free_webmails = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com", "mail.com"}
    executive_keywords = ["ceo", "cfo", "president", "hr", "human resources", "billing", "finance", "security", "support", "admin", "system", "verify", "verification", "login", "alert", "service"]
    
    audit_results = {
        "status": "Safe",
        "issues": []
    }
    
    core_domain = extract_core_domain(email_address)
    domain_sld = core_domain.split('.')[0] if '.' in core_domain else core_domain
    
    # Gibberish/DGA Domain Check
    if is_gibberish_domain(domain_sld):
        audit_results["issues"].append(f"🚨 Suspicious Sender Domain: The domain '{core_domain}' contains highly randomized character patterns (possible DGA spoofing).")
        audit_results["status"] = "Danger"
        
    # Sender Identity Mismatch Check (Display Name vs Address mismatch)
    if display_name and not check_sender_name_address_match(display_name, email_address):
        audit_results["issues"].append(f"⚠️ Sender Identity Mismatch: Display name '{display_name}' has no lexical correlation or visual overlap with the sending email address '{email_address}'. This is a common spoofing tactic.")
        if audit_results["status"] == "Safe":
            audit_results["status"] = "Suspicious"
            
    # Email Authentication Protocol DNS Checks (SPF/DMARC)
    auth_res = check_email_auth_records(domain)
    if auth_res["warnings"]:
        for warning in auth_res["warnings"]:
            audit_results["issues"].append(warning)
        if auth_res["spf"] == "Missing" or auth_res["dmarc"] == "Missing":
            if audit_results["status"] == "Safe":
                audit_results["status"] = "Suspicious"
        
    # Typosquatting check
    sld_normalized = clean_homoglyphs(domain_sld)
    for brand, off_domain in official_brands.items():
        brand_sld = off_domain.split('.')[0]
        brand_normalized = clean_homoglyphs(brand_sld)
        
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

    # Display Name check
    display_lower = display_name.lower()
    for brand, off_domain in official_brands.items():
        if brand in display_lower:
            if core_domain != off_domain and not core_domain.endswith("." + off_domain):
                if not any(brand in issue for issue in audit_results["issues"]):
                    audit_results["issues"].append(f"🚨 Display Name Spoofing: Display name claims to represent '{brand.capitalize()}' but actual sending domain is '{core_domain}' (expected: {off_domain}).")
                    audit_results["status"] = "Danger"
                    break
                    
    # Free Webmail Abuse check
    is_free_mail = core_domain in free_webmails
    if is_free_mail:
        has_exec_keyword = any(keyword in display_lower for keyword in executive_keywords)
        has_brand_keyword = any(brand in display_lower for brand in official_brands)
        if has_exec_keyword or has_brand_keyword:
            audit_results["issues"].append(f"🚨 Free Webmail Corporate Abuse: Official corporate or brand claims originating from a public email provider ({domain}). Real services use custom enterprise domains.")
            audit_results["status"] = "Danger"
            
    # Executive Username check
    username_lower = username.lower()
    if not is_free_mail and audit_results["status"] == "Safe":
        for keyword in ["ceo", "cfo", "president"]:
            if keyword == username_lower or username_lower.startswith(keyword + "-") or username_lower.startswith(keyword + "_"):
                audit_results["issues"].append(f"⚠️ Executive Identity Flag: Sender username starts with high-priority role '{keyword.upper()}'. Validate via alternative contact channel.")
                audit_results["status"] = "Suspicious"
                
    # Reply-To Mismatch and Gibberish check
    if reply_to_address and reply_to_address.strip():
        reply_to_clean = reply_to_address.strip()
        if "@" in reply_to_clean:
            reply_parts = reply_to_clean.split("@")
            if len(reply_parts) == 2:
                reply_core_domain = extract_core_domain(reply_to_clean)
                reply_domain_sld = reply_core_domain.split('.')[0] if '.' in reply_core_domain else reply_core_domain
                
                # Check for DGA on Reply-To Domain
                if is_gibberish_domain(reply_domain_sld):
                    audit_results["issues"].append(f"🚨 Suspicious Reply-To Domain: The Reply-To domain '{reply_core_domain}' contains highly randomized character patterns (possible DGA spoofing).")
                    audit_results["status"] = "Danger"
                
                if reply_core_domain != core_domain:
                    audit_results["issues"].append(f"🚨 Reply-To Header Mismatch: Reply-To address '{reply_to_clean}' points to a different core domain than the From address '{email_address}'. Replies will be routed to an external domain.")
                    audit_results["status"] = "Danger"
                
    # Mailed-By Domain Alignment (via) check
    if mailed_by and mailed_by.strip():
        mailed_by_clean = mailed_by.strip().lower()
        mailed_core = extract_core_domain(mailed_by_clean)
        if mailed_core != core_domain:
            audit_results["issues"].append(f"⚠️ Sender Domain Alignment Warning: Email sent via '{mailed_by_clean}' which does not match the From domain '{core_domain}' (displayed as 'via' in Gmail). This indicates mailing relay or possible spoofing.")
            if audit_results["status"] == "Safe":
                audit_results["status"] = "Suspicious"
                
    return audit_results

def check_google_safe_browsing(url, api_key):
    if not api_key:
        return {"is_malicious": False, "threat_types": []}
        
    parse_url = url
    if not parse_url.startswith(("http://", "https://")):
        parse_url = "http://" + parse_url
        
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload = {
        "client": {
            "clientId": "phishguard-ai",
            "clientVersion": "1.0.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [
                {"url": parse_url}
            ]
        }
    }
    print("[Google Safe Browsing] API working...")
    try:
        import requests
        response = requests.post(endpoint, json=payload, timeout=4)
        if response.status_code == 200:
            res = response.json()
            if "matches" in res:
                threats = [match.get("threatType", "UNKNOWN") for match in res["matches"]]
                return {"is_malicious": True, "threat_types": threats}
        return {"is_malicious": False, "threat_types": []}
    except Exception:
        return {"is_malicious": False, "threat_types": []}

def scan_urls_in_text(text):
    url_pattern = r'https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:/\S*)?'
    raw_urls = re.findall(url_pattern, text)
    if not raw_urls:
        return []
        
    audit_results = []
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
    shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"]
    
    # Load local threat database to check links too
    intel_data = load_local_intel()
    blacklisted_domains_lower = [d.lower() for d in intel_data.get("blacklisted_domains", [])]
    
    seen_urls = set()
    for url in raw_urls:
        url_clean = url.strip("[]()\"' ,.;:-")
        if not url_clean or url_clean in seen_urls:
            continue
        if "." not in url_clean or len(url_clean) < 4:
            continue
        seen_urls.add(url_clean)
        
        analysis = {
            "url": url_clean,
            "status": "Safe",
            "issues": []
        }
        
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
            
        # Threat Intel override check
        core_domain = extract_core_domain(url_clean)
        if core_domain.lower() in blacklisted_domains_lower or hostname.lower() in blacklisted_domains_lower:
            analysis["issues"].append("🚨 Local Threat Intelligence: Target domain is manually blacklisted by user.")
            analysis["status"] = "Danger"
            
        if url_clean.startswith("http://"):
            analysis["issues"].append("❌ Insecure Protocol: Uses 'http://' instead of secure 'https://'.")
            if analysis["status"] != "Danger":
                analysis["status"] = "Suspicious"
            
        if any(shortener in hostname for shortener in shorteners):
            analysis["issues"].append("⚠️ Obfuscated Link: Uses a URL shortener service which hides the destination.")
            if analysis["status"] != "Danger":
                analysis["status"] = "Suspicious"
            
        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        if re.match(ip_pattern, hostname):
            analysis["issues"].append("🚨 Severe Risk: Uses a raw IP address as a domain, avoiding registration checks.")
            analysis["status"] = "Danger"
            
        for brand, domains in official_brands.items():
            if brand in hostname:
                is_official = any(off_domain in hostname for off_domain in domains)
                if not is_official:
                    analysis["issues"].append(f"🚨 Brand Spoofing: Contains brand keyword '{brand.capitalize()}' but core domain resolves to '{hostname}' instead of official '{domains[0]}'.")
                    analysis["status"] = "Danger"
                    break
                    
        if hostname.count(".") >= 4:
            analysis["issues"].append("⚠️ Domain Structure: Excessive number of subdomains (common in credential harvesting spoofing).")
            if analysis["status"] != "Danger":
                analysis["status"] = "Suspicious"
                
        if url_model is not None and url_vectorizer is not None:
            try:
                lex_feats = np.array([get_url_lexical_features(url_clean)])
                tfidf_feats = url_vectorizer.transform([url_clean])
                combined_feats = sp.hstack([lex_feats, tfidf_feats], format='csr')
                probs = url_model.predict_proba(combined_feats)[0]
                pred_class_idx = np.argmax(probs)
                pred_prob = probs[pred_class_idx]
                class_labels = ['benign', 'phishing', 'defacement', 'malware']
                pred_label = class_labels[pred_class_idx]
                
                if pred_label != 'benign' and pred_prob > 0.6:
                    analysis["issues"].append(f"🤖 Machine Learning Alert: URL classified as '{pred_label.upper()}' with {pred_prob*100:.1f}% confidence.")
                    analysis["status"] = "Danger"
            except Exception:
                pass
                
        # Check Google Safe Browsing API
        if GOOGLE_SAFE_BROWSING_API_KEY:
            gsb_res = check_google_safe_browsing(url_clean, GOOGLE_SAFE_BROWSING_API_KEY)
            if gsb_res["is_malicious"]:
                threat_labels = ", ".join(gsb_res["threat_types"])
                analysis["issues"].append(f"🚨 Google Safe Browsing Alert: URL flagged as malicious ({threat_labels}).")
                analysis["status"] = "Danger"
            else:
                analysis["issues"].append("✅ Google Safe Browsing: Verified Clean (Not listed on global threat blocklists).")
                
        audit_results.append(analysis)
    return audit_results

# Asynchronous Retraining Pipeline Task
def retrain_models_task():
    global model, vectorizer
    print("Background Job: Initiating PhishGuard AI retraining pipeline...")
    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        
        # 1. Load baseline dataset
        baseline_path = os.path.join("dataset", "enron_spam_data.csv")
        if not os.path.exists(baseline_path):
            print("Retraining aborting: Base Enron file dataset/enron_spam_data.csv is missing.")
            return
            
        print("Loading baseline Enron dataset...")
        df_base = pd.read_csv(baseline_path)
        df_base['text'] = df_base['Subject'].fillna('') + ' ' + df_base['Message'].fillna('')
        df_base = df_base[['text', 'Spam/Ham']].dropna()
        df_base['label'] = df_base['Spam/Ham'].map({'ham': 0, 'spam': 1})
        df_base = df_base[['text', 'label']]
        
        # 2. Load feedback dataset
        df_feedback = None
        if os.path.exists(FEEDBACK_CSV_PATH) and os.path.getsize(FEEDBACK_CSV_PATH) > 0:
            try:
                print("Loading feedback correction dataset...")
                df_feedback = pd.read_csv(FEEDBACK_CSV_PATH)
                df_feedback = df_feedback[['text', 'label']].dropna()
                df_feedback['label'] = df_feedback['label'].astype(int)
            except Exception as e:
                print(f"Feedback CSV reading advisory: {e}")
                
        # 3. Merge datasets
        if df_feedback is not None and len(df_feedback) > 0:
            print(f"Combining datasets. Baseline: {len(df_base)}, User Feedback: {len(df_feedback)}")
            df_combined = pd.concat([df_base, df_feedback], ignore_index=True)
        else:
            df_combined = df_base
            
        df_combined = df_combined.drop_duplicates().reset_index(drop=True)
        
        # 4. Cleaning
        print("Vector Preprocessing text (this takes a brief moment)...")
        df_combined['cleaned_text'] = df_combined['text'].apply(preprocess_text)
        
        X = df_combined['cleaned_text']
        y = df_combined['label']
        
        # 5. Fit TfidfVectorizer
        print("Extracting TF-IDF features...")
        new_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        X_tfidf = new_vectorizer.fit_transform(X)
        
        # 6. Fit LinearSVC Classifier
        print("Fitting Linear Support Vector Classifier...")
        new_model = LinearSVC(C=1.0, random_state=42)
        new_model.fit(X_tfidf, y)
        
        # 7. Write checkpoints
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(new_model, f)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(new_vectorizer, f)
            
        # 8. Reload in memory
        model = new_model
        vectorizer = new_vectorizer
        print("Background Job Completed: Models updated successfully!")
    except Exception as e:
        print(f"Background Job Error: Retraining failed: {e}")

# Pydantic Schemas for Requests & Responses
class ScanRequest(BaseModel):
    text: str = Field(..., description="The full email text body to scan")
    sender_name: Optional[str] = Field("", description="The sender display name")
    sender_email: Optional[str] = Field("", description="The sender actual email address")
    reply_to: Optional[str] = Field("", description="The reply-to header address")
    mailed_by: Optional[str] = Field("", description="The mailed-by/relay domain (via)")

class UrlRequest(BaseModel):
    url: str = Field(..., description="A single URL to inspect")

class FeedbackRequest(BaseModel):
    text: str = Field(..., description="The email body text")
    sender_name: Optional[str] = Field("", description="The sender display name")
    sender_email: Optional[str] = Field("", description="The sender actual email address")
    reply_to: Optional[str] = Field("", description="The reply-to header address")
    user_label: int = Field(..., description="The correct label: 0 for safe (ham), 1 for phishing (spam)")

@app.get("/")
def read_root():
    # Show stats on local intel and feedback
    intel = load_local_intel()
    feedback_count = 0
    if os.path.exists(FEEDBACK_CSV_PATH) and os.path.getsize(FEEDBACK_CSV_PATH) > 0:
        try:
            with open(FEEDBACK_CSV_PATH, "r", encoding="utf-8") as f:
                feedback_count = max(0, sum(1 for line in f) - 1)
        except Exception:
            pass
            
    device_status = "CPU"
    if HAS_BERT:
        import torch
        device_status = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        
    return {
        "status": "online",
        "app": "PhishGuard AI API",
        "computation_device": device_status,
        "models_loaded": {
            "svm_email_classifier": model is not None,
            "url_xgb_classifier": url_model is not None,
            "bert_dl_classifier": bert_pipeline is not None
        },
        "adaptive_intelligence": {
            "feedback_samples_collected": feedback_count,
            "local_blacklisted_domains": len(intel.get("blacklisted_domains", [])),
            "local_blacklisted_emails": len(intel.get("blacklisted_emails", []))
        }
    }

@app.post("/api/scan-email")
def scan_email(req: ScanRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Email models not loaded on backend server.")
        
    text = req.text
    sender_name = req.sender_name or ""
    sender_email = req.sender_email or ""
    reply_to = req.reply_to or ""
    mailed_by = req.mailed_by or ""
    
    # 0. Audit Headers if provided
    header_audit = None
    if sender_email.strip():
        header_audit = audit_email_headers(sender_name, sender_email, reply_to, mailed_by)
        
    # Local Threat Intel Check (Zero-Day Blacklisting)
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
        
    # 1. Run SVM
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred_svm = int(model.predict(vectorized)[0])
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized)[0]
        prob_svm = float(probs[1])
    elif hasattr(model, "decision_function"):
        decision_score = model.decision_function(vectorized)[0]
        prob_svm = float(1 / (1 + np.exp(-decision_score)))
    else:
        prob_svm = 0.5
        
    # Apply rules overrides (same as app.py)
    text_lower = text.lower()
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
    
    if lottery_spam or phishing_spam_urgent or phishing_spam_soft_svm:
        pred_svm = 1
        prob_svm = max(prob_svm, 0.95)
        
    # 2. Run BERT if available (GPU accelerated)
    pred_bert = None
    prob_bert = None
    if bert_pipeline is not None:
        try:
            truncated_text = text[:1000]
            res = bert_pipeline(truncated_text)[0]
            pred_bert = 1 if res['label'] == 'LABEL_1' else 0
            prob_bert = float(res['score'])
            if pred_bert == 0:
                prob_bert = 1.0 - prob_bert
                
            phishing_spam_soft_bert = (
                any(w in text_lower for w in soft_words) and 
                any(w in text_lower for w in action_words) and 
                has_link_instruction and 
                pred_bert == 1
            )
            if lottery_spam or phishing_spam_urgent or phishing_spam_soft_bert:
                pred_bert = 1
                prob_bert = max(prob_bert, 0.95)
        except Exception:
            pass
            
    # 3. Dynamic URL Scanning (includes checking local blacklists)
    url_audit_results = scan_urls_in_text(text)
    
    # 4. Stacking Model Fusion
    if bert_pipeline is not None and prob_bert is not None:
        ensemble_score = (prob_svm * 0.4) + (prob_bert * 0.6)
    else:
        ensemble_score = prob_svm
        
    ensemble_verdict_idx = 1 if ensemble_score > 0.5 else 0
    
    # 5. Consensus Verdict logic based on Stacking Fusion
    verdict = "Safe"
    if ensemble_verdict_idx == 1:
        verdict = "Danger" if ensemble_score > 0.75 else "Suspicious"
        
    # Elevate if headers audit or links or local blacklists are danger
    if local_blacklist_triggered:
        verdict = "Danger"
    elif header_audit and header_audit["status"] == "Danger":
        verdict = "Danger"
    elif header_audit and header_audit["status"] == "Suspicious" and verdict == "Safe":
        verdict = "Suspicious"
        
    if any(audit["status"] == "Danger" for audit in url_audit_results):
        verdict = "Danger"
    elif any(audit["status"] == "Suspicious" for audit in url_audit_results) and verdict == "Safe":
        verdict = "Suspicious"
        
    # Detailed Consensus message using Ensemble calculations
    if local_blacklist_triggered:
        consensus_msg = "🚨 CRITICAL THREAT: Local threat database has flagged the sender domain or email address."
    elif any(audit["status"] == "Danger" for audit in url_audit_results):
        consensus_msg = f"🚨 DANGER: Malicious URL detected in email body (Text Fusion: {(1.0 - ensemble_score)*100 if ensemble_verdict_idx == 0 else ensemble_score*100:.1f}%)"
    elif header_audit and header_audit["status"] == "Danger":
        consensus_msg = f"🚨 DANGER: High-risk sender spoofing detected (Text Fusion: {(1.0 - ensemble_score)*100 if ensemble_verdict_idx == 0 else ensemble_score*100:.1f}%)"
    elif any(audit["status"] == "Suspicious" for audit in url_audit_results):
        consensus_msg = f"⚠️ SUSPICIOUS: Potential threat links detected (Text Fusion: {(1.0 - ensemble_score)*100 if ensemble_verdict_idx == 0 else ensemble_score*100:.1f}%)"
    elif header_audit and header_audit["status"] == "Suspicious":
        consensus_msg = f"⚠️ SUSPICIOUS: Mismatch in sender alignment (Text Fusion: {(1.0 - ensemble_score)*100 if ensemble_verdict_idx == 0 else ensemble_score*100:.1f}%)"
    elif bert_pipeline is not None and prob_bert is not None:
        if ensemble_verdict_idx == 1:
            consensus_msg = f"HIGH-CONFIDENCE PHISHING ALERT (Ensemble Fusion Score: {ensemble_score*100:.1f}%)"
        else:
            consensus_msg = f"SAFE EMAIL (Ensemble Fusion Score: {(1.0 - ensemble_score)*100:.1f}%)"
    else:
        if ensemble_verdict_idx == 1:
            consensus_msg = f"PHISHING ALERT DETECTED (Traditional SVM Classifier flagged spam patterns, Fusion: {ensemble_score*100:.1f}%)"
        else:
            consensus_msg = f"SAFE EMAIL (Traditional SVM Classifier verified safety, Fusion: {(1.0 - ensemble_score)*100:.1f}%)"
 
    return {
        "verdict": verdict,
        "consensus_verdict_details": consensus_msg,
        "svm": {
            "prediction": pred_svm,
            "confidence": prob_svm if pred_svm == 1 else (1.0 - prob_svm)
        },
        "bert": {
            "available": bert_pipeline is not None,
            "prediction": pred_bert,
            "confidence": prob_bert if pred_bert == 1 else (1.0 - prob_bert) if prob_bert is not None else None
        },
        "ensemble": {
            "score": ensemble_score,
            "verdict": "Spam / Phishing" if ensemble_verdict_idx == 1 else "Legitimate (Ham)",
            "confidence": ensemble_score if ensemble_verdict_idx == 1 else (1.0 - ensemble_score)
        },
        "header_audit": header_audit,
        "url_audit": url_audit_results
    }

@app.post("/api/scan-url")
def scan_url(req: UrlRequest):
    if url_model is None or url_vectorizer is None:
        raise HTTPException(status_code=503, detail="URL models not loaded on backend server.")
    url_audit_results = scan_urls_in_text(req.url)
    if not url_audit_results:
        url_audit_results = [{
            "url": req.url,
            "status": "Suspicious",
            "issues": ["⚠️ Unrecognized URL format. Could not establish domain validation checks."]
        }]
    return url_audit_results[0]

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    text = req.text.strip()
    sender_name = req.sender_name or ""
    sender_email = req.sender_email or ""
    reply_to = req.reply_to or ""
    user_label = req.user_label
    
    if not text:
        raise HTTPException(status_code=400, detail="Text body is required to record feedback.")
        
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
            
    # 3. Add task to background_tasks to run the model retraining
    background_tasks.add_task(retrain_models_task)
    
    return {
        "status": "success",
        "message": "Feedback registered successfully.",
        "intel_updated": intel_updated,
        "retraining_started": True
    }

@app.post("/api/retrain")
def run_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_models_task)
    return {"status": "success", "message": "Model retraining initiated in the background."}

# WSGI adapter wrapper for compatibility with WSGI-only servers (like PythonAnywhere)
try:
    from a2wsgi import ASGIMiddleware
    wsgi_app = ASGIMiddleware(app)
except ImportError:
    wsgi_app = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
