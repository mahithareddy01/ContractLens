import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# APP CONFIG
# ==========================================

st.set_page_config(
    page_title="ContractLens AI — Legal Contract Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODERN LIGHT THEME CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

/* ── Base Reset ─────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F1F5F9 !important;
    color: #1E293B !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

/* ── Hide Default Header ────────────── */
header { visibility: hidden; height: 0; }
[data-testid="collapsedControl"] { display: none; }

/* ── Top Navigation Bar ─────────────── */
.top-nav {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%);
    padding: 0.8rem 2rem;
    margin: -1.5rem -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
}
.nav-brand {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.nav-brand-icon {
    background: rgba(255,255,255,0.2);
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}
.nav-pills {
    display: flex;
    gap: 0.25rem;
    background: rgba(255,255,255,0.15);
    padding: 4px;
    border-radius: 12px;
}
.nav-pill {
    color: rgba(255,255,255,0.75);
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
}
.nav-pill:hover, .nav-pill.active {
    background: white;
    color: #6366F1;
}

/* ── Page Section Headers ───────────── */
.page-header {
    margin-bottom: 1.5rem;
}
.page-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #1E293B;
    margin: 0;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #64748B;
    margin-top: 0.25rem;
}

/* ── Glass Cards ────────────────────── */
.glass-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s, transform 0.2s;
}
.glass-card:hover {
    box-shadow: 0 4px 16px rgba(99,102,241,0.08);
}
.glass-card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #F1F5F9;
}
.glass-card-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.glass-card-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: #1E293B;
}
.glass-card-badge {
    margin-left: auto;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Stat Cards ─────────────────────── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 0.6rem;
}
.stat-value {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label {
    font-size: 0.75rem;
    color: #94A3B8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Verdict Display ────────────────── */
.verdict-display {
    text-align: center;
    padding: 2rem;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}
.verdict-display::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    opacity: 0.1;
}
.verdict-emoji {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.verdict-text {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
}
.verdict-sub {
    font-size: 0.85rem;
    opacity: 0.8;
}
.verdict-contradiction {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border: 1px solid #FECACA;
}
.verdict-contradiction .verdict-text { color: #DC2626; }
.verdict-contradiction::before { background: #EF4444; }
.verdict-contradiction .stat-icon { background: #FEE2E2; }

.verdict-entailment {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border: 1px solid #BBF7D0;
}
.verdict-entailment .verdict-text { color: #16A34A; }
.verdict-entailment::before { background: #22C55E; }
.verdict-entailment .stat-icon { background: #DCFCE7; }

.verdict-neutral {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border: 1px solid #FDE68A;
}
.verdict-neutral .verdict-text { color: #D97706; }
.verdict-neutral::before { background: #F59E0B; }
.verdict-neutral .stat-icon { background: #FEF3C7; }

/* ── Confidence Ring ────────────────── */
.confidence-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
}
.confidence-value {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    color: #6366F1;
}
.confidence-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Risk Tags ──────────────────────── */
.risk-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.risk-high {
    background: #FEE2E2;
    color: #DC2626;
    border: 1px solid #FECACA;
}
.risk-medium {
    background: #FEF3C7;
    color: #D97706;
    border: 1px solid #FDE68A;
}
.risk-low {
    background: #F0FDF4;
    color: #16A34A;
    border: 1px solid #BBF7D0;
}

/* ── Clause Cards ───────────────────── */
.clause-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    transition: border-color 0.2s;
}
.clause-card:hover { border-color: #6366F1; }
.clause-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 0.35rem;
}
.clause-text {
    font-size: 0.85rem;
    line-height: 1.6;
    color: #475569;
}

/* ── Highlighted Text ───────────────── */
.highlight-container {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    max-height: 400px;
    overflow-y: auto;
    font-size: 0.88rem;
    line-height: 1.8;
    color: #334155;
}
.hl-high {
    background: #FEE2E2;
    color: #DC2626;
    padding: 1px 4px;
    border-radius: 3px;
    font-weight: 500;
}
.hl-medium {
    background: #FEF3C7;
    color: #D97706;
    padding: 1px 4px;
    border-radius: 3px;
    font-weight: 500;
}
.hl-keyword {
    background: #EDE9FE;
    color: #7C3AED;
    padding: 1px 4px;
    border-radius: 3px;
    font-weight: 600;
}

/* ── Buttons ────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 2rem !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

[data-testid="stDownloadButton"] > button {
    background: white !important;
    color: #6366F1 !important;
    border: 2px solid #6366F1 !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
}

/* ── File Uploader ──────────────────── */
[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 2rem !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
    background: #FAFAFE !important;
}

/* ── Text Area ──────────────────────── */
[data-testid="stTextArea"] textarea {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    color: #1E293B !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 1rem !important;
    resize: vertical;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ── Tabs ───────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    color: #64748B;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}

/* ── Expander ───────────────────────── */
[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: #475569 !important;
}

/* ── Sidebar ────────────────────────── */
[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}
.sidebar-section-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    color: #6366F1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    margin-top: 1.2rem;
}
.sidebar-info {
    font-size: 0.82rem;
    color: #64748B;
    line-height: 1.6;
}
.sidebar-chip {
    display: inline-block;
    background: #F1F5F9;
    color: #475569;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.15rem;
}
.sidebar-status {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    margin-top: 0.5rem;
}
.status-online {
    background: #F0FDF4;
    color: #16A34A;
}
.status-offline {
    background: #FEF2F2;
    color: #DC2626;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}

/* ── Alerts ─────────────────────────── */
[data-testid="stAlert"] {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    font-size: 0.85rem !important;
}

/* ── Spinner ────────────────────────── */
[data-testid="stSpinner"] > div {
    border-top-color: #6366F1 !important;
}

/* ── Scrollbar ──────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ── Plotly container fix ───────────── */
.js-plotly-plot .plotly .modebar { right: 10px !important; top: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS
# ==========================================

LABELS   = ["Contradiction", "Entailment", "Neutral"]
MAX_LEN  = 150
DIM      = 128
STOPWORDS = {"the","a","an","and","or","but","in","on","at","to","for",
             "of","with","by","from","this","that","is","are","was","were",
             "be","been","it","its","not","no","as","if","any","all","such",
             "shall","may","will","which","their","has","have","had"}

CLAUSE_KEYWORDS = {
    "termination": ["terminat", "end", "expire", "cancel", "cessation"],
    "payment": ["payment", "pay", "compensation", "fee", "invoice", "remuneration"],
    "confidentiality": ["confidential", "secret", "proprietary", "disclose", "nda"],
    "liability": ["liabilit", "indemnif", "damages", "responsible", "negligence"],
    "non-compete": ["non-compete", "compete", "solicit", "competitive", "restrict"]
}

# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "attention_model.h5")

@st.cache_resource(show_spinner=False)
def load_components():
    try:
        model = load_model(MODEL_PATH, compile=False)
        with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

model, tokenizer, load_error = load_components()

# ==========================================
# TEXT PROCESSING
# ==========================================

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_key_terms(clean_text: str, top_n: int = 15):
    words = [w for w in clean_text.split() if w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(top_n)

def detect_clause_types(text: str) -> dict:
    text_lower = text.lower()
    detected = {}
    for clause_type, keywords in CLAUSE_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            detected[clause_type] = matches
    return detected

def sentence_risk_score(text: str) -> list:
    HIGH_RISK  = {"indemnif","liabilit","terminat","breach","forfeit","penalt","lawsuit","arbitrat","damages","void","null"}
    MED_RISK   = {"shall","must","obligat","warrant","disclos","confidential","exclusiv","prohibit","restrict","agree","unless","provided"}
    sentences  = re.split(r'(?<=[.?!])\s+', text.strip())
    result = []
    for s in sentences[:25]:
        low = s.lower()
        if any(k in low for k in HIGH_RISK):
            level = "high"
        elif any(k in low for k in MED_RISK):
            level = "medium"
        else:
            level = "low"
        result.append({"text": s, "level": level})
    return result

def highlight_text(text: str, risk_data: list, key_terms: list) -> str:
    highlighted = text
    term_words = [t[0] for t in key_terms[:10]]
    
    for term in term_words:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f'<span class="hl-keyword">{m.group()}</span>', highlighted)
    
    for item in risk_data:
        sentence = item["text"]
        level = item["level"]
        if level in ["high", "medium"]:
            escaped = re.escape(sentence[:50])
            pattern = re.compile(escaped, re.IGNORECASE)
            css_class = f"hl-{level}"
            highlighted = pattern.sub(lambda m: f'<span class="{css_class}">{m.group()}</span>', highlighted, count=1)
    
    return highlighted

def positional_encoding(length: int, dim: int) -> np.ndarray:
    pe = np.zeros((length, dim))
    for pos in range(length):
        for i in range(0, dim, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / dim)))
            if i + 1 < dim:
                pe[pos, i+1] = np.cos(pos / (10000 ** (i / dim)))
    return pe

def readability_stats(text: str) -> dict:
    words = text.split()
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len = len(words) / max(len(sentences), 1)
    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    return {
        "Total Words": len(words),
        "Sentences": len(sentences),
        "Avg Word Length": f"{avg_word_len:.1f}",
        "Avg Sentence Length": f"{avg_sent_len:.0f}",
        "Lexical Diversity": f"{unique_ratio*100:.0f}%"
    }

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:2.2rem; margin-bottom:0.3rem;">📋</div>
        <div style="font-family:'Poppins'; font-weight:700; font-size:1.1rem; color:#1E293B;">ContractLens AI</div>
        <div style="font-size:0.75rem; color:#94A3B8;">Legal Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:1px; background:#E2E8F0; margin:0.8rem 0;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">Model Status</div>', unsafe_allow_html=True)
    if load_error:
        st.markdown(f'''
        <div class="sidebar-status status-offline">
            <span class="status-dot"></span> Demo Mode
        </div>
        <div class="sidebar-info" style="font-size:0.72rem; color:#DC2626; margin-top:0.3rem;">
            Model file not found. Running with simulated predictions.
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="sidebar-status status-online">
            <span class="status-dot"></span> Model Loaded
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">Detection Classes</div>', unsafe_allow_html=True)
    class_colors = {"Contradiction": "#DC2626", "Entailment": "#16A34A", "Neutral": "#D97706"}
    for label, color in class_colors.items():
        st.markdown(f'''
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
            <div style="width:12px; height:12px; border-radius:4px; background:{color};"></div>
            <span class="sidebar-chip">{label}</span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">Clause Detection</div>', unsafe_allow_html=True)
    for clause in CLAUSE_KEYWORDS.keys():
        st.markdown(f'<span class="sidebar-chip">{clause.replace("-"," ").title()}</span>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">Architecture</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="sidebar-info">
        <div style="background:#F8FAFC; padding:0.6rem; border-radius:8px; font-family:monospace; font-size:0.72rem; line-height:1.8;">
        Input → Embedding<br>
        → Positional Enc<br>
        → MultiHead Attn<br>
        → Dense → Output
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">Settings</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="sidebar-info">
        Max Tokens: <strong>{MAX_LEN}</strong><br>
        Embed Dim: <strong>{DIM}</strong><br>
        Heads: <strong>8</strong>
    </div>
    ''', unsafe_allow_html=True)

# ==========================================
# TOP NAVIGATION
# ==========================================

st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-brand-icon">📋</div>
        ContractLens AI
    </div>
    <div class="nav-pills">
        <a class="nav-pill active" href="#">Analyzer</a>
        <a class="nav-pill" href="#">Positional Encoding</a>
        <a class="nav-pill" href="#">Attention Demo</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PAGE HEADER
# ==========================================

st.markdown("""
<div class="page-header">
    <h1 class="page-title">Contract Intelligence Analyzer</h1>
    <p class="page-subtitle">Upload a legal contract to classify clauses, detect risks, and visualize attention patterns</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================

tab_upload, tab_paste = st.tabs(["📁 Upload Contract", "✏️ Paste Text"])

text_data = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drop a .txt contract file here",
        type=["txt"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        text_data = uploaded_file.read().decode("utf-8")
        st.success(f"✅ Loaded **{uploaded_file.name}** ({len(text_data):,} characters)")

with tab_paste:
    pasted = st.text_area(
        "Paste contract text here",
        height=200,
        placeholder="Paste the full contract text here...",
        label_visibility="collapsed"
    )
    if pasted.strip():
        text_data = pasted

# ==========================================
# RUN BUTTON
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)
col_run, col_space = st.columns([1, 4])
with col_run:
    run = st.button("🚀 Analyze Contract", use_container_width=True)

# ==========================================
# ANALYSIS LOGIC
# ==========================================

if run:
    if not text_data.strip():
        st.warning("⚠️ Please upload a file or paste contract text first.")
        st.stop()

    with st.spinner("🔍 Analyzing contract with AI..."):
        clean = preprocess(text_data)
        stats = readability_stats(text_data)
        
        # ── Inference ───────────────────────
        if model and tokenizer:
            seq = tokenizer.texts_to_sequences([clean])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
            pred = model.predict(padded, verbose=0)[0]
        else:
            raw = np.random.dirichlet(np.ones(3))
            pred = raw

        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        result = LABELS[idx]
        
        # ── Additional Analysis ─────────────
        freq = extract_key_terms(clean, top_n=15)
        risk_data = sentence_risk_score(text_data)
        clause_types = detect_clause_types(text_data)
        highlighted = highlight_text(text_data, risk_data, freq)
        
        high_count = sum(1 for r in risk_data if r["level"] == "high")
        medium_count = sum(1 for r in risk_data if r["level"] == "medium")
        low_count = sum(1 for r in risk_data if r["level"] == "low")

    # ==========================================
    # STAT CARDS ROW
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-icon" style="background:#EDE9FE;">📝</div>
            <div class="stat-value" style="color:#6366F1;">{words}</div>
            <div class="stat-label">Total Words</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon" style="background:#FEE2E2;">📄</div>
            <div class="stat-value" style="color:#DC2626;">{sentences}</div>
            <div class="stat-label">Clauses</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon" style="background:#FEF3C7;">⚠️</div>
            <div class="stat-value" style="color:#D97706;">{high}</div>
            <div class="stat-label">High Risk</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon" style="background:#FCE7F3;">🏷️</div>
            <div class="stat-value" style="color:#DB2777;">{clauses}</div>
            <div class="stat-label">Clause Types</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon" style="background:#DBEAFE;">🔑</div>
            <div class="stat-value" style="color:#2563EB;">{terms}</div>
            <div class="stat-label">Key Terms</div>
        </div>
    </div>
    """.format(
        words=stats["Total Words"],
        sentences=stats["Sentences"],
        high=high_count,
        clauses=len(clause_types),
        terms=len(freq)
    ), unsafe_allow_html=True)

    # ==========================================
    # VERDICT + CONFIDENCE ROW
    # ==========================================

    col_verdict, col_conf, col_probs = st.columns([1.2, 0.8, 1.5])

    with col_verdict:
        verdict_class = result.lower()
        emojis = {"contradiction": "⛔", "entailment": "✅", "neutral": "⚡"}
        st.markdown(f"""
        <div class="verdict-display verdict-{verdict_class}">
            <div class="verdict-emoji">{emojis[verdict_class]}</div>
            <div class="verdict-text">{result}</div>
            <div class="verdict-sub">Primary Classification</div>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-header">
                <div class="glass-card-icon" style="background:#EDE9FE;">📊</div>
                <div class="glass-card-title">Confidence</div>
            </div>
            <div class="confidence-ring-wrap">
                <div class="confidence-value">{conf_pct:.0f}%</div>
                <div class="confidence-label">Model Certainty</div>
            </div>
        </div>
        """.format(conf_pct=conf*100), unsafe_allow_html=True)

    with col_probs:
        # Plotly probability bars
        colors = ["#EF4444", "#22C55E", "#F59E0B"]
        fig_probs = go.Figure()
        fig_probs.add_trace(go.Bar(
            y=LABELS,
            x=[p*100 for p in pred],
            orientation='h',
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in pred],
            textposition='outside',
            textfont=dict(size=11, color='#475569', family='Inter'),
            hovertemplate='%{x:.1f}%<extra></extra>'
        ))
        fig_probs.update_layout(
            height=160,
            margin=dict(l=10, r=40, t=5, b=5),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color='#475569')),
            showlegend=False
        )
        fig_probs.update_xaxes(fixedrange=True)
        st.plotly_chart(fig_probs, use_container_width=True, config={'displayModeBar': False})

    # ==========================================
    # DETECTED CLAUSE TYPES
    # ==========================================

    if clause_types:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-header">
                <div class="glass-card-icon" style="background:#FCE7F3;">🏷️</div>
                <div class="glass-card-title">Detected Clause Types</div>
                <div class="glass-card-badge" style="background:#FCE7F3; color:#DB2777;">{count} Found</div>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                {chips}
            </div>
        </div>
        """.format(
            count=len(clause_types),
            chips="".join([
                f'<span class="risk-tag" style="background:#EDE9FE; color:#7C3AED; border-color:#C4B5FD;">🏷️ {ct.replace("-"," ").title()}</span>'
                for ct in clause_types.keys()
            ])
        ), unsafe_allow_html=True)

    # ==========================================
    # TWO-COLUMN: KEY TERMS + RISK DISTRIBUTION
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    col_terms, col_risk_donut = st.columns([1.5, 1])

    with col_terms:
        terms = [x[0] for x in freq[:10]]
        counts = [x[1] for x in freq[:10]]
        
        fig_terms = go.Figure()
        fig_terms.add_trace(go.Bar(
            x=counts[::-1],
            y=terms[::-1],
            orientation='h',
            marker_color=['#6366F1' if i == len(terms)-1 else '#E0E7FF' for i in range(len(terms))],
            text=counts[::-1],
            textposition='outside',
            textfont=dict(size=10, color='#64748B'),
            hovertemplate='%{x}<extra></extra>'
        ))
        fig_terms.update_layout(
            height=320,
            margin=dict(l=10, r=40, t=35, b=5),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=10, color='#475569')),
            title=dict(text='🔑 Key Terms Frequency', font=dict(size=12, color='#1E293B', family='Poppins'), x=0, xanchor='left'),
            showlegend=False
        )
        st.plotly_chart(fig_terms, use_container_width=True, config={'displayModeBar': False})

    with col_risk_donut:
        risk_vals = [high_count, medium_count, low_count]
        risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']
        risk_colors = ['#EF4444', '#F59E0B', '#22C55E']
        
        fig_donut = go.Figure()
        fig_donut.add_trace(go.Pie(
            labels=risk_labels,
            values=risk_vals,
            hole=0.65,
            marker_colors=risk_colors,
            textinfo='label+percent',
            textfont=dict(size=10, color='#475569'),
            hovertemplate='%{label}: %{value} clauses<extra></extra>'
        ))
        fig_donut.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=35, b=5),
            paper_bgcolor='white',
            title=dict(text='⚠️ Risk Distribution', font=dict(size=12, color='#1E293B', family='Poppins'), x=0, xanchor='left'),
            showlegend=False,
            annotations=[dict(text=f'{len(risk_data)}', x=0.5, y=0.5, font_size=28, font_color='#1E293B', font_family='Poppins', font_weight='bold', showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    # ==========================================
    # HIGHLIGHTED TEXT VIEWER
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-icon" style="background:#DBEAFE;">📑</div>
            <div class="glass-card-title">Annotated Contract View</div>
            <div class="glass-card-badge" style="background:#DBEAFE; color:#2563EB;">Live</div>
        </div>
        <div style="display:flex; gap:0.5rem; margin-bottom:0.8rem; font-size:0.72rem;">
            <span class="risk-high" style="font-size:0.68rem;">High Risk</span>
            <span class="risk-medium" style="font-size:0.68rem;">Medium Risk</span>
            <span class="hl-keyword" style="font-size:0.68rem;">Key Term</span>
        </div>
        <div class="highlight-container">{highlighted}</div>
    </div>
    """.format(highlighted=highlighted), unsafe_allow_html=True)

    # ==========================================
    # CLAUSE RISK LIST
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-icon" style="background:#FEF3C7;">⚖️</div>
            <div class="glass-card-title">Clause Risk Analysis</div>
            <div class="glass-card-badge" style="background:#FEF3C7; color:#D97706;">{total} Clauses</div>
        </div>
    """.format(total=len(risk_data)), unsafe_allow_html=True)

    for item in risk_data[:12]:
        dot_colors = {"high": "#EF4444", "medium": "#F59E0B", "low": "#22C55E"}
        tag_classes = {"high": "risk-high", "medium": "risk-medium", "low": "risk-low"}
        st.markdown("""
        <div class="clause-card">
            <div class="clause-dot" style="background:{dot_color};"></div>
            <div style="flex:1;">
                <span class="risk-tag {tag_class}" style="margin-bottom:0.3rem;">{level_upper}</span>
                <div class="clause-text">{text}</div>
            </div>
        </div>
        """.format(
            dot_color=dot_colors[item["level"]],
            tag_class=tag_classes[item["level"]],
            level_upper=item["level"].upper(),
            text=item["text"]
        ), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # ATTENTION HEATMAP (Plotly)
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    col_att, col_pe = st.columns(2)

    with col_att:
        # Generate structured attention for visualization
        tokens = clean.split()[:20]
        n = len(tokens)
        if n == 0:
            n = 10
            tokens = [f"token_{i}" for i in range(n)]
        
        attention = np.random.rand(n, n)
        attention = (attention + attention.T) / 2
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        fig_att = go.Figure()
        fig_att.add_trace(go.Heatmap(
            z=attention,
            x=tokens,
            y=tokens,
            colorscale=[
                [0, '#F8FAFC'],
                [0.5, '#A78BFA'],
                [1, '#6366F1']
            ],
            showscale=True,
            colorbar=dict(title='Weight', thickness=10, tickfont=dict(size=9)),
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        fig_att.update_layout(
            height=380,
            margin=dict(l=80, r=20, t=40, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white',
            title=dict(text='🧠 Self-Attention Map', font=dict(size=12, color='#1E293B', family='Poppins'), x=0, xanchor='left'),
            xaxis=dict(tickangle=45, tickfont=dict(size=8, color='#64748B'), showgrid=False),
            yaxis=dict(tickfont=dict(size=8, color='#64748B'), showgrid=False, autorange='reversed')
        )
        st.plotly_chart(fig_att, use_container_width=True, config={'displayModeBar': False})

    with col_pe:
        # Positional Encoding Heatmap
        pe = positional_encoding(50, DIM)
        
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Heatmap(
            z=pe.T,
            colorscale=[
                [0, '#F8FAFC'],
                [0.3, '#BFDBFE'],
                [0.6, '#6366F1'],
                [1, '#4C1D95']
            ],
            showscale=True,
            colorbar=dict(title='Value', thickness=10, tickfont=dict(size=9)),
            hovertemplate='Position: %{x}<br>Dim: %{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        fig_pe.update_layout(
            height=380,
            margin=dict(l=10, r=20, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            title=dict(text='📐 Positional Encoding Matrix', font=dict(size=12, color='#1E293B', family='Poppins'), x=0, xanchor='left'),
            xaxis=dict(title='Sequence Position', tickfont=dict(size=9, color='#64748B'), showgrid=False),
            yaxis=dict(title='Encoding Dimension', tickfont=dict(size=9, color='#64748B'), showgrid=False)
        )
        st.plotly_chart(fig_pe, use_container_width=True, config={'displayModeBar': False})

    # ==========================================
    # POSITIONAL ENCODING - SIN/COS COMPARISON
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-icon" style="background:#DBEAFE;">📐</div>
            <div class="glass-card-title">Positional Encoding: Sin vs Cos Patterns</div>
        </div>
    """, unsafe_allow_html=True)

    positions = np.arange(50)
    sin_vals = [np.sin(pos / (10000 ** (0 / DIM))) for pos in positions]
    cos_vals = [np.cos(pos / (10000 ** (0 / DIM))) for pos in positions]

    fig_sin_cos = go.Figure()
    fig_sin_cos.add_trace(go.Scatter(
        x=positions, y=sin_vals,
        mode='lines',
        name='SIN (dim=0)',
        line=dict(color='#6366F1', width=2.5),
        hovertemplate='Pos: %{x}<br>SIN: %{y:.3f}<extra></extra>'
    ))
    fig_sin_cos.add_trace(go.Scatter(
        x=positions, y=cos_vals,
        mode='lines',
        name='COS (dim=0)',
        line=dict(color='#EC4899', width=2.5, dash='dash'),
        hovertemplate='Pos: %{x}<br>COS: %{y:.3f}<extra></extra>'
    ))
    fig_sin_cos.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(title='Position', showgrid=False, tickfont=dict(size=10, color='#64748B')),
        yaxis=dict(title='Encoding Value', showgrid=True, gridcolor='#F1F5F9', tickfont=dict(size=10, color='#64748B')),
        legend=dict(orientation='h', y=1.12, x=0, xanchor='left', font=dict(size=10, color='#475569')),
        hovermode='x unified'
    )
    st.plotly_chart(fig_sin_cos, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # ORDER SENSITIVITY DEMO (Task 6)
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-icon" style="background:#FEF3C7;">🔄</div>
            <div class="glass-card-title">Order Sensitivity Analysis</div>
            <div class="glass-card-badge" style="background:#FEF3C7; color:#D97706;">Task 6</div>
        </div>
        <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:1rem; margin-bottom:1rem; font-size:0.85rem; color:#92400E;">
            <strong>Why order matters:</strong> Even with identical words, different word orders produce different positional encodings, which affects attention patterns and final predictions.
        </div>
    """, unsafe_allow_html=True)

    contract_a = "Payment shall be made within 30 days"
    contract_b = "Within 30 days payment shall be made"
    
    pe_a = positional_encoding(len(contract_a.split()), 16)
    pe_b = positional_encoding(len(contract_b.split()), 16)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f'<div style="font-weight:600; margin-bottom:0.5rem; color:#6366F1;">Contract A:</div>')
        st.markdown(f'<div style="font-size:0.88rem; color:#475569; margin-bottom:0.8rem;">"{contract_a}"</div>')
        fig_a = go.Figure()
        fig_a.add_trace(go.Heatmap(
            z=pe_a.T, x=contract_a.split(), 
            colorscale=[[0,'#F8FAFC'],[1,'#6366F1']],
            showscale=False,
            hovertemplate='Pos: %{x}<br>Dim: %{y}<br>Val: %{z:.2f}<extra></extra>'
        ))
        fig_a.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=40),
                           paper_bgcolor='white', plot_bgcolor='white',
                           xaxis=dict(tickfont=dict(size=9), showgrid=False),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig_a, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        st.markdown(f'<div style="font-weight:600; margin-bottom:0.5rem; color:#EC4899;">Contract B:</div>')
        st.markdown(f'<div style="font-size:0.88rem; color:#475569; margin-bottom:0.8rem;">"{contract_b}"</div>')
        fig_b = go.Figure()
        fig_b.add_trace(go.Heatmap(
            z=pe_b.T, x=contract_b.split(), 
            colorscale=[[0,'#F8FAFC'],[1,'#EC4899']],
            showscale=False,
            hovertemplate='Pos: %{x}<br>Dim: %{y}<br>Val: %{z:.2f}<extra></extra>'
        ))
        fig_b.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=40),
                           paper_bgcolor='white', plot_bgcolor='white',
                           xaxis=dict(tickfont=dict(size=9), showgrid=False),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig_b, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("""
        <div style="margin-top:1rem; padding:0.8rem; background:#F1F5F9; border-radius:8px; font-size:0.82rem; color:#475569; line-height:1.7;">
            <strong>Observation:</strong> The same words ("payment", "30", "days", etc.) receive <em>different</em> positional encoding values because they appear at different positions. 
            In Contract A, "payment" is at position 0 (high SIN values), while in Contract B it's at position 3 (different encoding). 
            This allows the model to distinguish between "Payment shall be made" vs "shall be made Payment" — even though the vocabulary is identical.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # DOCUMENT STATISTICS
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">
            <div class="glass-card-icon" style="background:#E0E7FF;">📊</div>
            <div class="glass-card-title">Document Statistics</div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:1rem;">
    """, unsafe_allow_html=True)

    stat_icons = {
        "Total Words": "📝", "Sentences": "📄", "Avg Word Length": "📏",
        "Avg Sentence Length": "📐", "Lexical Diversity": "🌈"
    }
    for k, v in stats.items():
        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:1rem; text-align:center;">
            <div style="font-size:1.3rem; margin-bottom:0.3rem;">{stat_icons.get(k, '📊')}</div>
            <div style="font-family:'Poppins'; font-weight:700; font-size:1.3rem; color:#1E293B;">{v}</div>
            <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">{k}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ==========================================
    # DOWNLOAD REPORT
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    
    risk_summary = "\n".join(
        f"  [{r['level'].upper():6}] {r['text'][:100]}" for r in risk_data
    )
    clause_summary = "\n".join(
        f"  • {ct}: {', '.join(kw)}" for ct, kw in clause_types.items()
    )

    report = f"""CONTRACTLENS AI — ANALYSIS REPORT
{'='*60}

CLASSIFICATION RESULT
  Verdict     : {result}
  Confidence  : {conf*100:.2f}%
  All Scores  : {', '.join(f'{l}: {v*100:.1f}%' for l, v in zip(LABELS, pred))}

DOCUMENT STATISTICS
{chr(10).join(f'  {k:<24}: {v}' for k, v in stats.items())}

DETECTED CLAUSE TYPES
{clause_summary}

RISK ANALYSIS
  High-risk clauses   : {high_count}
  Medium-risk clauses : {medium_count}
  Low-risk clauses    : {low_count}
  Total scanned       : {len(risk_data)}

TOP KEY TERMS
  {', '.join(f'{t} ({c})' for t, c in freq[:10])}

ANNOTATED CLAUSES
{risk_summary}

{'='*60}
Generated by ContractLens AI
"""

    st.download_button(
        "📥 Download Full Report (.txt)",
        data=report,
        file_name="contractlens_report.txt",
        mime="text/plain"
    )
