# Climate Change - GIEC Chat (AR6)

Assistant RAG interrogeable sur le rapport du GIEC AR6 avec citations vérifiables.

## Arborescence
```
.
├── data/
│   ├── raw/              # PDFs AR6 (à télécharger)
│   └── processed/        # chunks_chapter1.json
├── vectorstore/          # index FAISS
├── ingest.py             # PDF -> chunks
├── vectorize.py          # chunks -> FAISS
├── app.py                # UI Streamlit
└── requirements.txt
```

## Installation
```bash
pip install -r requirements.txt
cp .env.example .env  # ajoute ta clé GROQ_API_KEY (https://console.groq.com/keys)
```

## Usage - Notebook (Colab)
1. Ouvre `Climate Change - GIEC chat.ipynb` dans Colab
2. Cell 4 : installe depuis `requirements.txt` (`%pip install -r requirements.txt`)
3. Cell 6 : ingestion full 1-12+SPM (auto-download depuis https://www.ipcc.ch/report/ar6/wg1/downloads/report/)
4. Cell 7 : vectorisation `BAAI/bge-m3` -> `vectorstore/giec_wg1_all`
5. Cell 12 : LLM Groq `llama-3.3-70b-versatile` (stream=True, clé via `.env` ou `userdata.get("GROQ_API_KEY")`, fallback `llama-3.1-8b-instant`)

## Usage - Webserver (Streamlit)
```bash
pip install -r requirements.txt
cp .env.example .env  # GROQ_API_KEY=gsk_xxx
streamlit run app.py  # http://localhost:8501
```
`app.py` est standalone : FAISS unifié + bge-m3 + Groq streaming + citations `[Source: WG1 Ch.X, p.Y]`.

## LLM Groq Cloud (gratuit)
- Modèle primaire : `llama-3.3-70b-versatile` (131k ctx, 280 tps), fallback `llama-3.1-8b-instant` sur 429/quota
- Clé gratuite sans CB : https://console.groq.com/keys -> `.env` ou `st.secrets["GROQ_API_KEY"]`
- Env : `GROQ_API_KEY`, optionnel `GROQ_MODEL_PRIMARY/FALLBACK`

## Roadmap
- [x] Semaine 1: Ingestion Ch1 (pypdf + decouper)
- [x] Semaine 2: RAG + LLM Groq avec citations (stream)
- [ ] Semaine 3: Module Fact-Check + Dashboard NLP

## Pipeline
PDF du GIEC
    ↓
Extraction texte
    ↓
Chunking
    ↓
~1341 chunks
    ↓
BAAI/bge-m3
    ↓
1341 embeddings
    ↓
Vector database
    ↓
Question utilisateur
    ↓
Embedding de la question
    ↓
Recherche des chunks similaires
    ↓
Top-k chunks
    ↓
LLM
    ↓
Réponse