"""
GIEC Chat - AR6 WG1 (1-12+SPM) - App Streamlit avec Groq Cloud
Deploy: streamlit run app.py
Env: .env avec GROQ_API_KEY (https://console.groq.com/keys)
Vectorstore: vectorstore/giec_wg1_all (fallback giec_wg1_ch1)
"""
import os
import json
from pathlib import Path

import faiss
import numpy as np
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

st.set_page_config(page_title="GIEC Chat - AR6 WG1 (1-12+SPM)", layout="wide")
st.title("🌍 GIEC Chat - AR6 WG1 (Ch.1-12 + SPM)")
st.caption("RAG unifié FAISS + BAAI/bge-m3 + Groq groq/compound (stream)")

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
    st.warning("Index unifié non trouvé. Lance les cellules 4 et 7 (full 1-12+SPM) pour générer `vectorstore/giec_wg1_all`.")
    st.info("PDFs auto-téléchargés depuis: https://www.ipcc.ch/report/ar6/wg1/downloads/report/")
    st.stop()

# Groq client
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else os.getenv("GROQ_API_KEY")
if not groq_key:
    st.warning("GROQ_API_KEY manquant. Ajoute-le dans `.env` ou `st.secrets` (https://console.groq.com/keys).")
    st.info("Local: crée un fichier `.env` avec `GROQ_API_KEY=...` à côté de `app.py`. Streamlit Cloud: Settings → Secrets.")
    groq_key = st.text_input("Entre ta clé Groq (ne sera pas stockée)", type="password")
    if not groq_key:
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_key

from groq import Groq
client = Groq()

def detect_lang(q: str) -> str:
    fr_markers = ["quelle","quel","pourquoi","comment","est-ce","réchauffement","chapitre","où","combien"]
    return "fr" if any(w in q.lower() for w in fr_markers) else "en"

def rechercher(query: str, k: int = 5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q_emb, dtype="float32"), k=k)
    docs = [(texts[i], metadatas[i], float(scores[0][rank])) for rank, i in enumerate(ids[0])]
    return docs

def repondre_stream(query: str, k: int = 5, temperature: float = 0):
    docs = rechercher(query, k=k)
    contexte = "\n\n".join(f"[Source: {m['source']}, p.{m['page']}]\\n{t[:1500]}" for t, m, _ in docs)
    lang = detect_lang(query)
    system_prompt = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Contexte:\\n{contexte}\\n\\nQuestion: {query}"}
    ]
    def call(model):
        return client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=800, stream=True)
    try:
        stream = call(GROQ_MODEL_PRIMARY)
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "quota" in msg:
            st.warning(f"Quota {GROQ_MODEL_PRIMARY} atteint -> fallback {GROQ_MODEL_FALLBACK}")
            stream = call(GROQ_MODEL_FALLBACK)
        else:
            raise
    return stream, docs

# UI
chapters = sorted(set(m["chapter"] for m in metadatas), key=lambda x: (isinstance(x, str), x))
sel = st.selectbox("Filtrer par chapitre", ["Tous"] + [f"Ch.{c}" for c in chapters])
k = st.slider("Top-k chunks", 3, 10, 5)
query = st.text_input("Pose ta question au rapport du GIEC :", placeholder="Ex: Quelle est la probabilité de dépasser 1.5°C avant 2040 ?")

if query:
    stream, docs = repondre_stream(query, k=k)
    # filtrage optionnel après recherche
    if sel != "Tous":
        docs = [d for d in docs if f"Ch.{d[1]['chapter']}" == sel]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Réponse")
        # Streaming
        placeholder = st.empty()
        full = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full += delta
                placeholder.markdown(full)
        if not full:
            st.warning("Aucune génération (vérifie ta clé Groq).")

    with col2:
        st.subheader("Sources vérifiables")
        for t, m, s in docs:
            with st.expander(f"Ch.{m['chapter']} p.{m['page']} [score={s:.3f}]"):
                st.caption(f"{m['source']} — {m.get('chapter_title','')}")
                st.write(t[:1000])
        st.caption("Généré par Groq groq/compound (stream) — citations obligatoires.")
