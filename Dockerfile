FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for faiss + build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .env.example ./
COPY vectorstore ./vectorstore
# data/processed is ignored (regenerable), but vectorstore is needed at runtime
# If you keep vectorstore on volume, comment next line and mount volume
# COPY vectorstore ./vectorstore

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
