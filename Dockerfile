# ==============================================================================
# Dockerfile para Despliegue en Azure Web App for Containers
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download es_core_news_sm && \
    python -m nltk.downloader stopwords

COPY . /app

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true"]
