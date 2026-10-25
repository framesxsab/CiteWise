FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG CITEWISE_DEFAULT_MODEL=e5-base-v2
ENV CITEWISE_DEFAULT_MODEL=${CITEWISE_DEFAULT_MODEL} CITEWISE_DEVICE=cpu \
    CITEWISE_DATA_DIR=/data/processed CITEWISE_EMBED_CACHE_DIR=/data/cache/embeddings \
    HF_HUB_OFFLINE=1 HF_HOME=/home/citewise/.cache/huggingface PORT=8080
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[models]' \
    && useradd --create-home --uid 1000 citewise \
    && mkdir -p /data/processed /data/cache/embeddings \
    && chown -R citewise:citewise /data /home/citewise
USER citewise
# Weights are pinned and staged at image build time; runtime Hugging Face access is disabled.
RUN python -m citewise.fetch_model --model "${CITEWISE_DEFAULT_MODEL}"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import json,os,urllib.request; url='http://127.0.0.1:'+os.getenv('PORT','8080')+'/api/health'; assert json.load(urllib.request.urlopen(url,timeout=2))['status']=='ready'" || exit 1
CMD ["sh", "-c", "uvicorn citewise.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
