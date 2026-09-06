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
```

## Usage
1. Télécharger WG1 Chapitre 1: https://www.ipcc.ch/report/ar6/wg1/
   -> Placer `IPCC_AR6_WGI_Chapter01.pdf` dans `data/raw/`
2. `python ingest.py`
3. `python vectorize.py`
4. `streamlit run app.py`

## Roadmap
- [ ] Semaine 1: Ingestion Ch1
- [ ] Semaine 2: RAG + LLM avec citations
- [ ] Semaine 3: Module Fact-Check + Dashboard NLP
