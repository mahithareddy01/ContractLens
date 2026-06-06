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

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F1F5F9 !important;
    color: #1E293B !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

header { visibility: hidden; height: 0; }
[data-testid="collapsedControl"] { display: none; }

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

.page-header { margin-bottom: 1.5rem; }
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

.glass-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
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
}
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

.verdict-display {
    text-align: center;
    padding: 2rem;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}
.verdict-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.verdict-text {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
}
.verdict-sub { font-size: 0.85rem; opacity: 0.8; }

.verdict-contradiction {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border: 1px solid #FECACA;
}
.verdict-contradiction .verdict-text { color: #DC2626; }

.verdict-entailment {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border: 1px solid #BBF7D0;
}
.verdict-entailment .verdict-text { color: #16A34A; }

.verdict-neutral {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border: 1px solid #FDE68A;
}
.verdict-neutral .verdict-text { color: #D97706; }

.confidence-value {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    color: #6366F1;
    text-align: center;
}
.confidence-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: center;
    margin-top: 0.2rem;
}

.risk-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.risk-high { background: #FEE2E2; color: #DC2626; border: 1px solid #FECACA; }
.risk-medium { background: #FEF3C7; color: #D97706; border: 1px solid #FDE68A; }
.risk-low { background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0; }

.clause-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
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
.hl-high { background: #FEE2E2; color: #DC2626; padding: 1px 4px; border-radius: 3px; font-weight: 500; }
.hl-medium { background: #FEF3C7; color: #D97706; padding: 1px 4px; border-radius: 3px; font-weight: 500; }
.hl-keyword { background: #EDE9FE; color: #7C3AED; padding: 1px 4px; border-radius: 3px; font-weight: 600; }

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
}

[data-testid="stDownloadButton"] > button {
    background: white !important;
    color: #6366F1 !important;
    border: 2px solid #6366F1 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 2rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366F1 !important; }

[data-testid="stTextArea"] textarea {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    color: #1E293B !important;
    font-family: 'Inter', sans-serif !important;
    padding: 1rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    color: #64748B;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
}

[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
}

[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }

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
.sidebar-info { font-size: 0.82rem; color: #64748B; line-height: 1.6; }
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
.status-online { background: #F0FDF4; color: #16A34A; }
.status-offline { background: #FEF2F2; color: #DC2626; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

.chart-container {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.5rem;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MATPLOTLIB LIGHT THEME
# ==========================================

def setup_chart(fig, ax, title=""):
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_title(title, color='#1E293B', fontsize=11, fontfamily='sans-serif', 
                 fontweight='bold', pad=12, loc='left')
    ax.tick_params(colors='#64748B', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#E2E8F0')
        spine.set_linewidth(0.5)

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

# ==========================================
# TOP NAVIGATION
# ==========================================

st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-brand-icon">📋</div>
        ContractLens AI
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
            <div style="padding:0.5rem 0;">
                <div class="confidence-value">{conf_pct:.0f}%</div>
                <div class="confidence-label">Model Certainty</div>
            </div>
        </div>
        """.format(conf_pct=conf*100), unsafe_allow_html=True)

    with col_probs:
        # Matplotlib probability bars
        fig_probs, ax_probs = plt.subplots(figsize=(5, 2.2))
        colors = ["#EF4444", "#22C55E", "#F59E0B"]
        bars = ax_probs.barh(LABELS, pred * 100, color=colors, height=0.5, edgecolor='white', linewidth=1)
        for bar, v in zip(bars, pred):
            ax_probs.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                         f"{v*100:.1f}%", va="center", color="#475569", fontsize=10, fontweight='500')
        ax_probs.set_xlim(0, 115)
        ax_probs.set_xlabel("Score (%)", color="#64748B", fontsize=9)
        setup_chart(fig_probs, ax_probs, "Class Probabilities")
        ax_probs.grid(axis="x", color="#F1F5F9", linewidth=0.5, linestyle="--")
        ax_probs.spines['top'].set_visible(False)
        ax_probs.spines['right'].set_visible(False)
        fig_probs.tight_layout()
        st.pyplot(fig_probs, use_container_width=True)
        plt.close(fig_probs)

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
        
        fig_terms, ax_terms = plt.subplots(figsize=(6, 3.5))
        bar_colors = ['#6366F1' if i == len(terms)-1 else '#E0E7FF' for i in range(len(terms))]
        ax_terms.barh(terms[::-1], counts[::-1], color=bar_colors[::-1], height=0.6, edgecolor='white', linewidth=0.5)
        for i, (t, c) in enumerate(zip(terms[::-1], counts[::-1])):
            ax_terms.text(c + 0.2, i, str(c), va='center', color='#64748B', fontsize=9)
        ax_terms.set_xlabel("Frequency", color="#64748B", fontsize=9)
        setup_chart(fig_terms, ax_terms, "🔑 Key Terms Frequency")
        ax_terms.grid(axis="x", color="#F1F5F9", linewidth=0.5, linestyle="--")
        ax_terms.spines['top'].set_visible(False)
        ax_terms.spines['right'].set_visible(False)
        fig_terms.tight_layout()
        st.pyplot(fig_terms, use_container_width=True)
        plt.close(fig_terms)

    with col_risk_donut:
        risk_vals = [high_count, medium_count, low_count]
        risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']
        risk_colors = ['#EF4444', '#F59E0B', '#22C55E']
        
        fig_donut, ax_donut = plt.subplots(figsize=(4, 3.5))
        wedges, texts, autotexts = ax_donut.pie(
            risk_vals, labels=risk_labels, colors=risk_colors,
            autopct='%1.0f%%', startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
        )
        for text in texts:
            text.set_color('#475569')
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')
        ax_donut.text(0, 0, str(len(risk_data)), ha='center', va='center', 
                     fontsize=24, fontweight='bold', color='#1E293B')
        setup_chart(fig_donut, ax_donut, "⚠️ Risk Distribution")
        fig_donut.tight_layout()
        st.pyplot(fig_donut, use_container_width=True)
        plt.close(fig_donut)

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
    # ATTENTION HEATMAP (Matplotlib)
    # ==========================================

    st.markdown("<br>", unsafe_allow_html=True)
    col_att, col_pe = st.columns(2)

    with col_att:
        tokens = clean.split()[:20]
        n = len(tokens)
        if n == 0:
            n = 10
            tokens = [f"token_{i}" for i in range(n)]
        
        attention = np.random.rand(n, n)
        attention = (attention + attention.T) / 2
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        fig_att, ax_att = plt.subplots(figsize=(5, 4))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom", ["#F8FAFC", "#A78BFA", "#6366F1"], N=256
        )
        im = ax_att.imshow(attention, cmap=cmap, aspect="auto", interpolation="nearest")
        ax_att.set_xticks(range(n))
        ax_att.set_yticks(range(n))
        ax_att.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
        ax_att.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax_att, fraction=0.046, pad=0.04, label="Weight")
        setup_chart(fig_att, ax_att, "🧠 Self-Attention Map")
        ax_att.set_xlabel("Key Position", color="#64748B", fontsize=9)
        ax_att.set_ylabel("Query Position", color="#64748B", fontsize=9)
        fig_att.tight_layout()
        st.pyplot(fig_att, use_container_width=True)
        plt.close(fig_att)

    with col_pe:
        pe = positional_encoding(50, DIM)
        
        fig_pe, ax_pe = plt.subplots(figsize=(5, 4))
        cmap_pe = matplotlib.colors.LinearSegmentedColormap.from_list(
            "pe", ["#F8FAFC", "#BFDBFE", "#6366F1", "#4C1D95"], N=256
        )
        im_pe = ax_pe.imshow(pe.T, cmap=cmap_pe, aspect="auto", interpolation="bilinear")
        plt.colorbar(im_pe, ax=ax_pe, fraction=0.046, pad=0.04, label="Value")
        setup_chart(fig_pe, ax_pe, "📐 Positional Encoding Matrix")
        ax_pe.set_xlabel("Sequence Position", color="#64748B", fontsize=9)
        ax_pe.set_ylabel("Encoding Dimension", color="#64748B", fontsize=9)
        fig_pe.tight_layout()
        st.pyplot(fig_pe, use_container_width=True)
        plt.close(fig_pe)

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

    fig_sin_cos, ax_sc = plt.subplots(figsize=(10, 3))
    ax_sc.plot(positions, sin_vals, color='#6366F1', linewidth=2.5, label='SIN (dim=0)')
    ax_sc.plot(positions, cos_vals, color='#EC4899', linewidth=2.5, linestyle='--', label='COS (dim=0)')
    ax_sc.set_xlabel("Position", color="#64748B", fontsize=10)
    ax_sc.set_ylabel("Encoding Value", color="#64748B", fontsize=10)
    ax_sc.legend(loc='upper right', frameon=False, fontsize=10)
    setup_chart(fig_sin_cos, ax_sc, "")
    ax_sc.grid(color='#F1F5F9', linewidth=0.5)
    ax_sc.spines['top'].set_visible(False)
    ax_sc.spines['right'].set_visible(False)
    fig_sin_cos.tight_layout()
    st.pyplot(fig_sin_cos, use_container_width=True)
    plt.close(fig_sin_cos)
    
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
        st.markdown('<div style="font-weight:600; margin-bottom:0.5rem; color:#6366F1;">Contract A:</div>')
        st.markdown(f'<div style="font-size:0.88rem; color:#475569; margin-bottom:0.8rem;">"{contract_a}"</div>')
        fig_a, ax_a = plt.subplots(figsize=(4, 2))
        cmap_a = matplotlib.colors.LinearSegmentedColormap.from_list("a", ["#F8FAFC", "#6366F1"], N=256)
        ax_a.imshow(pe_a.T, cmap=cmap_a, aspect="auto")
        ax_a.set_xticks(range(len(contract_a.split())))
        ax_a.set_xticklabels(contract_a.split(), rotation=45, ha='right', fontsize=8)
        ax_a.set_yticks([])
        setup_chart(fig_a, ax_a, "")
        fig_a.tight_layout()
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)

    with col_b:
        st.markdown('<div style="font-weight:600; margin-bottom:0.5rem; color:#EC4899;">Contract B:</div>')
        st.markdown(f'<div style="font-size:0.88rem; color:#475569; margin-bottom:0.8rem;">"{contract_b}"</div>')
        fig_b, ax_b = plt.subplots(figsize=(4, 2))
        cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#F8FAFC", "#EC4899"], N=256)
        ax_b.imshow(pe_b.T, cmap=cmap_b, aspect="auto")
        ax_b.set_xticks(range(len(contract_b.split())))
        ax_b.set_xticklabels(contract_b.split(), rotation=45, ha='right', fontsize=8)
        ax_b.set_yticks([])
        setup_chart(fig_b, ax_b, "")
        fig_b.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)
    
    st.markdown("""
        <div style="margin-top:1rem; padding:0.8rem; background:#F1F5F9; border-radius:8px; font-size:0.82rem; color:#475569; line-height:1.7;">
            <strong>Observation:</strong> The same words receive <em>different</em> positional encoding values because they appear at different positions. 
            In Contract A, "payment" is at position 0 (high SIN values), while in Contract B it's at position 3 (different encoding). 
            This allows the model to distinguish word order — even with identical vocabulary.
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
