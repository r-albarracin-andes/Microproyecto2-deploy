#!/bin/bash
# ==============================================================================
# Script de Inicio de la Aplicación en Azure App Service (Linux)
# ==============================================================================

echo "Iniciando configuración del entorno en Azure App Service..."

# Cambiar al directorio donde reside este script y la aplicación
cd "$(dirname "$0")"

# Asegurar la disponibilidad del modelo spaCy y recursos NLTK si no se cargaron durante build
python -m spacy download es_core_news_sm || true
python -m nltk.downloader stopwords || true

echo "Lanzando servidor Streamlit en el puerto 8000..."
# Azure App Service Linux redirige el tráfico HTTP entrante al puerto 8000 por defecto
python -m streamlit run app.py \
    --server.port 8000 \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.serverAddress 0.0.0.0 \
    --server.headless true
