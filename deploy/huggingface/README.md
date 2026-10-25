---
title: CiteWise
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8080
---

# CiteWise deployment notes

This metadata is a template for a separate Hugging Face Docker Space. It does not publish the project or prove an account can create a dynamic Space. Current Hugging Face documentation lists CPU Basic at 2 vCPU / 16 GB RAM with no hourly compute charge, while new Docker/Gradio Spaces require a paid plan; upgraded hardware is billed while running. Confirm account eligibility and current terms in the Space settings before creation.

## Build and runtime configuration

Copy the project `Dockerfile`, `pyproject.toml`, `src/`, and project README to the Space repository root. Set the build variable `CITEWISE_DEFAULT_MODEL=e5-base-v2` (or another pinned supported base model) before the image is built. The Docker build stages the selected model checkpoint at its immutable revision. The app has `HF_HUB_OFFLINE=1` at runtime and loads one model per Uvicorn worker. Keep one worker for CPU memory.

Attach or stage the prepared QASPER validation file at `/data/processed/validation.jsonl`, then set `CITEWISE_DATA_DIR=/data/processed`. Attach a read-write Storage Bucket at `/data/cache` for the precomputed vectors and set `CITEWISE_EMBED_CACHE_DIR=/data/cache/embeddings`. Space root disk is ephemeral; the bucket keeps vectors across restarts. The validation file is real QASPER data under CC BY 4.0; retain its attribution and source revision in the published Space description.

Precompute the selected model embeddings before serving traffic, using the same pinned revision and data file:

```sh
python -m citewise.prepare_embeddings --model e5-base-v2 --split validation --data-dir /data/processed --cache-dir /data/cache/embeddings --output /data/cache/embedding-precompute.json
```

The command performs local inference and can take substantial CPU time. Save its JSON report and verify the resulting files are present before enabling public traffic. Changing the dataset, text normalization, model revision, or embedding settings requires a new cache and report.

## Local Docker check

From the project root:

```sh
docker build --build-arg CITEWISE_DEFAULT_MODEL=e5-base-v2 -t citewise:e5-base-v2 .
docker run --rm -p 8080:8080 -e CITEWISE_DATA_DIR=/data/processed -e CITEWISE_EMBED_CACHE_DIR=/data/cache/embeddings -v "${PWD}/data/processed:/data/processed:ro" -v citewise-cache:/data/cache citewise:e5-base-v2
```

Then verify `/api/health`, load the page, select a paper/question, and compare the five displayed evidence passages to `/api/search`. A local container result is not public HTTPS deployment evidence. Before calling it deployed, capture the public Space URL and verify health, model inference, HTTPS, desktop/mobile, keyboard behavior, and axe results there.
