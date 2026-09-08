"""
GIEC Chat — AR6 WG1 (1-12+SPM) — Interface graphique distinctive
Design: glaciology lab notebook — paper + cyan glacial + heat amber
Deploy: streamlit run app.py  |  Env: .env GROQ_API_KEY
Vectorstore: vectorstore/giec_wg1_all (fallback giec_wg1_ch1)
"""
import os, json
from pathlib import Path
import faiss, numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

VECTORSTORE_PATH = Path("vectorstore/giec_wg1_all")
if not (VECTORSTORE_PATH / "index.faiss").exists():
    VECTORSTORE_PATH = Path("vectorstore/giec_wg1_ch1")
EMBEDDING_MODEL = "BAAI/bge-m3"
GROQ_MODEL_PRIMARY = "groq/compound"
GROQ_MODEL_FALLBACK = "qwen/qwen3.8-27b"
SYSTEM_PROMPT_FR = """Tu es l'assistant GIEC. Réponds UNIQUEMENT avec le contexte fourni. Cite obligatoirement [Source: WG1 Ch.X/SPM, p.Y]. Si la réponse n'est pas dans le contexte, dis exactement "Non trouvé dans le rapport". N'invente rien. Réponds en français."""
SYSTEM_PROMPT_EN = """You are the IPCC assistant. Answer ONLY from the provided context. Cite [Source: WG1 Ch.X/SPM, p.Y]. If not in context, say exactly "Not found in the report". Do not hallucinate. Answer in English."""

st.set_page_config(page_title="GIEC Chat — AR6 WG1", layout="wide", initial_sidebar_state="collapsed")

# --- Design tokens + distinctive lab notebook identity
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,soft,wght@9..144,50..100,600;9..144,50..100,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#FCFCF9; --ink:#0F172A; --muted:#64748B; --line:#E2E8F0;
  --glacial:#0E7490; --glacial-2:#155E75; --glacial-light:#ECFEFF;
  --ember:#EA580C; --ember-light:#FFF7ED;
}
html, body, [data-testid="stAppViewContainer"]{ background:var(--paper); color:var(--ink); }
[data-testid="stHeader"]{ background:transparent; }
h1,h2,h3{ font-family:"Fraunces", serif !important; letter-spacing:-0.02em; }
p, label, div{ font-family:"Inter", system-ui, sans-serif; }
.mono{ font-family:"JetBrains Mono", monospace; }

/* Top bar */
.topbar{ display:flex; justify-content:space-between; align-items:center; padding:10px 0 14px; border-bottom:1px solid var(--line); margin-bottom:18px;}
.badge{ display:inline-flex; gap:8px; align-items:center; font-family:"JetBrains Mono", monospace; font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); }
.badge b{ background:var(--ink); color:var(--paper); padding:4px 8px; border-radius:999px; font-weight:600; letter-spacing:0.02em; }

/* Hero — spine + query */
.hero{ display:grid; grid-template-columns: 300px 1fr; gap:28px; align-items:stretch; margin:18px 0 22px; }
@media (max-width: 900px){ .hero{ grid-template-columns:1fr; } }
.spine{ background:var(--ink); color:var(--paper); border-radius:16px; padding:18px 16px; position:relative; overflow:hidden; }
.spine:before{ content:""; position:absolute; inset:0; background: radial-gradient(600px 300px at 30% 10%, rgba(14,116,144,0.35), transparent 60%); pointer-events:none; }
.spine h3{ color:var(--paper); margin:0 0 12px; font-size:18px; }
.chapter-row{ display:flex; align-items:center; gap:10px; font-family:"JetBrains Mono", monospace; font-size:11px; color:#CBD5E1; padding:6px 8px; border-radius:8px; }
.chapter-row.active{ background:rgba(255,255,255,0.08); color:var(--paper); }
.chapter-row .dot{ width:22px; height:22px; border-radius:999px; display:grid; place-items:center; font-size:10px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.15);}
.chapter-row.active .dot{ background:var(--glacial); color:white; border-color:transparent; }
.query-card{ background:white; border:1px solid var(--line); border-radius:16px; padding:22px; box-shadow: 0 8px 24px rgba(15,23,42,0.06); }
.query-card h2{ font-size:28px; line-height:1.05; margin:0 0 8px; }
.query-card p{ color:var(--muted); margin:0 0 14px; font-size:14px; }
.pills{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }
.pill{ font-family:"JetBrains Mono", monospace; font-size:11px; background:var(--glacial-light); color:var(--glacial-2); border:1px solid #CFFAFE; padding:6px 10px; border-radius:999px; cursor:pointer; }

/* Answer paper */
.answer-wrap{ background:white; border:1px solid var(--line); border-radius:16px; overflow:hidden; box-shadow: 0 10px 30px rgba(15,23,42,0.07); }
.answer-head{ display:flex; justify-content:space-between; align-items:center; padding:14px 18px; background:var(--glacial-light); border-bottom:1px solid #CFFAFE; font-family:"JetBrains Mono", monospace; font-size:11px; letter-spacing:0.06em; text-transform:uppercase; color:var(--glacial-2); }
.answer-body{ padding:18px 20px; font-size:16px; line-height:1.6; }
.answer-body em{ background:var(--ember-light); padding:1px 4px; border-radius:4px; font-style:normal; border:1px solid #FFEDD5; }

/* Sources — perforated lab tape */
.sources-head{ font-family:"JetBrains Mono", monospace; font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); margin:6px 0 10px; }
.source-card{ background:white; border:1px solid var(--line); border-radius:12px; padding:14px 14px 14px 18px; position:relative; margin-bottom:10px; box-shadow: 0 4px 14px rgba(15,23,42,0.04);}
.source-card:before{ content:""; position:absolute; left:0; top:8px; bottom:8px; width:0; border-left:2px dotted #CBD5E1; }
.source-meta{ font-family:"JetBrains Mono", monospace; font-size:11px; color:var(--glacial-2); display:flex; gap:8px; align-items:center; }
.stamp{ background:var(--ink); color:white; font-family:"JetBrains Mono", monospace; font-size:10px; padding:3px 7px; border-radius:999px; }
.stream-dot{ width:8px; height:8px; border-radius:999px; background:var(--ember); display:inline-block; animation: pulse 1.2s infinite; }
@keyframes pulse{ 0%{ opacity:1; transform:scale(1)} 50%{ opacity:0.6; transform:scale(0.85)} 100%{ opacity:1; transform:scale(1)} }
@media (prefers-reduced-motion: reduce){ .stream-dot{ animation:none; } }

/* Streamlit widget tuning */
[data-testid="stTextInput"] input{ border-radius:12px !important; border:1px solid var(--line) !important; padding:12px 14px !important; font-size:15px !important; }
[data-testid="stSelectbox"] > div, [data-testid="stSlider"]{ background:white; border-radius:12px; }
footer{ visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# --- Load DB
@st.cache_resource
def load_db():
    if not (VECTORSTORE_PATH / "index.faiss").exists():
        return None, None, None, None
    index = faiss.read_index(str(VECTORSTORE_PATH / "index.faiss"))
    with open(VECTORSTORE_PATH / "chunks.json", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    metadatas = [d["metadata"] for d in data]
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, texts, metadatas, model

index, texts, metadatas, embedder = load_db()
if index is None:
    st.warning("Index unifié non trouvé. Lance les cellules 4 et 7 (full 1-12+SPM) pour générer vectorstore/giec_wg1_all.")
    st.stop()

# Groq
groq_key = os.getenv("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets else None)
if not groq_key:
    st.info("Ajoute ta clé Groq dans `.env` (GROQ_API_KEY) ou dans Streamlit Secrets. https://console.groq.com/keys")
    groq_key = st.text_input("Clé Groq", type="password", placeholder="gsk_...")
    if not groq_key:
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_key
from groq import Groq
client = Groq()

def detect_lang(q: str) -> str:
    fr = ["quelle","quel","pourquoi","comment","réchauffement","chapitre","où","combien","est-ce"]
    return "fr" if any(w in q.lower() for w in fr) else "en"

def rechercher(query: str, k: int = 5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q_emb, dtype="float32"), k=k)
    return [(texts[i], metadatas[i], float(scores[0][rank])) for rank, i in enumerate(ids[0])]

def repondre_stream(query: str, k: int = 5, temperature: float = 0):
    docs = rechercher(query, k=k)
    lang = detect_lang(query)
    system_prompt = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN
    def build_messages(docs_sub, max_chars=1000):
        ctx = "\n\n".join(f"[Source: {m['source']}, p.{m['page']}]\n{t[:max_chars]}" for t, m, _ in docs_sub)
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Contexte:\n{ctx}\n\nQuestion: {query}"}]
    max_chars = 1000 if k <= 5 else 800
    messages = build_messages(docs, max_chars)
    if sum(len(m["content"]) for m in messages) > 4500:
        docs = docs[:3]
        messages = build_messages(docs, 800)
    def call(model, msgs):
        return client.chat.completions.create(model=model, messages=msgs, temperature=temperature, max_tokens=600, stream=True)
    try:
        stream = call(GROQ_MODEL_PRIMARY, messages)
    except Exception as e:
        msg = str(e).lower()
        if "413" in msg or "request_too_large" in msg:
            docs = docs[:3]
            msgs = build_messages(docs, 800)
            try:
                stream = call(GROQ_MODEL_PRIMARY, msgs)
            except Exception as e2:
                stream = call(GROQ_MODEL_FALLBACK, msgs)
            return stream, docs
        elif "429" in msg or "rate" in msg:
            stream = call(GROQ_MODEL_FALLBACK, messages)
        else:
            raise
    return stream, docs

# --- Header
st.markdown("""
<div class="topbar">
  <div class="badge"><b>AR6 WG1</b> IPCC · 1–12 + SPM · 17 741 passages · BAAI/bge-m3 + FAISS</div>
  <div class="badge mono" style="gap:6px">FR/EN auto <span style="background:var(--glacial); color:white; padding:4px 8px; border-radius:999px;">Groq · groq/compound · stream</span></div>
</div>
""", unsafe_allow_html=True)

# --- Hero
chapters_sorted = sorted(set(m["chapter"] for m in metadatas), key=lambda x: (isinstance(x, str), x))
# Build spine html
spine_rows = ""
for ch in chapters_sorted[:12]:
    label = f"Ch.{ch}" if isinstance(ch, int) else str(ch)
    active = "active" if ch == 1 else ""
    spine_rows += f'<div class="chapter-row {active}"><span class="dot">{label.replace("Ch.","")}</span><span>{label} — {metadatas[0].get("chapter_title","") if ch==1 else ""}</span></div>'
# Hero layout via columns for Streamlit widget placement but with styled spine
left, right = st.columns([1, 2.1], gap="large")
with left:
    st.markdown(f"""
<div class="spine">
  <h3>Rapport — Tranches</h3>
  <div style="display:grid; gap:2px; position:relative;">
    {spine_rows}
    <div class="chapter-row" style="color:#93C5FD; font-size:10px; margin-top:6px; border-top:1px solid rgba(255,255,255,0.12); padding-top:10px;">17741 vecteurs · 1024 dims</div>
  </div>
</div>
""", unsafe_allow_html=True)
with right:
    st.markdown("""
<div class="query-card">
  <h2>Interroge le rapport,<br/>pas ta mémoire.</h2>
  <p>Chaque réponse cite sa source — chapitre et page — et refuse d'inventer. Pose une question en français ou en anglais.</p>
</div>
""", unsafe_allow_html=True)
    # Controls
    c1, c2 = st.columns([1.3, 1])
    with c1:
        sel = st.selectbox("Chapitre", ["Tous"] + [f"Ch.{c}" for c in chapters_sorted], label_visibility="collapsed")
    with c2:
        k = st.slider("Top-k", 3, 10, 5, label_visibility="collapsed")
    # Example pills
    st.markdown('<div class="pills">', unsafe_allow_html=True)
    examples = [
        "Quelle est la part du réchauffement d'origine humaine ?",
        "Probability to exceed 1.5°C before 2040?",
        "Hausse du niveau de la mer d'ici 2100 ?",
    ]
    # Use session_state to allow pills to fill input
    if "q" not in st.session_state:
        st.session_state.q = ""
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex{i}"):
            st.session_state.q = ex
    st.markdown('</div>', unsafe_allow_html=True)
    query = st.text_input("Question", value=st.session_state.q, placeholder="Ex: Quelle est la probabilité de dépasser 1.5°C avant 2040 ?", label_visibility="collapsed")

# --- Answer
if query:
    stream, docs = repondre_stream(query, k=k)
    if sel != "Tous":
        docs = [d for d in docs if f"Ch.{d[1]['chapter']}" == sel]

    col_a, col_b = st.columns([1.9, 1.1], gap="large")
    with col_a:
        st.markdown('<div class="answer-wrap"><div class="answer-head"><span><span class="stream-dot"></span>&nbsp; Réponse en direct — citations obligatoires</span><span class="mono">groq/compound</span></div><div class="answer-body">', unsafe_allow_html=True)
        placeholder = st.empty()
        full = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full += delta
                placeholder.markdown(full)
        if not full:
            st.warning("Aucune génération — vérifie ta clé Groq.")
        st.markdown('</div></div>', unsafe_allow_html=True)
        st.caption("Si la réponse n'est pas dans le contexte, l'assistant répond exactement « Non trouvé dans le rapport ».")

    with col_b:
        st.markdown('<div class="sources-head">Sources vérifiables — ruban perforé</div>', unsafe_allow_html=True)
        for t, m, s in docs:
            title = m.get("chapter_title","")
            with st.container():
                st.markdown(f"""
<div class="source-card">
  <div class="source-meta"><span class="stamp">p.{m['page']}</span> Ch.{m['chapter']} · {m['source']} <span style="margin-left:auto; color:var(--muted);">score {s:.3f}</span></div>
  <div style="font-size:11px; color:var(--muted); margin:4px 0 8px;" class="mono">{title}</div>
  <div style="font-size:13px; line-height:1.5; color:var(--ink);">{t[:520].replace('<','&lt;')}…</div>
</div>
""", unsafe_allow_html=True)
        st.caption("Index unifié giec_wg1_all — app.py (stream + .env)")
else:
    st.markdown('<div style="text-align:center; color:var(--muted); padding:18px; font-size:13px;" class="mono">↑ Essaie un exemple ci-dessus ou pose ta propre question</div>', unsafe_allow_html=True)
