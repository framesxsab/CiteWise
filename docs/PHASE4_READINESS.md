# Phase 4 model and deployment readiness

## Model availability and protocol

E5-base-v2 and GTE-base are fallback comparison models, explicitly older than the originally targeted BGE-M3 and Qwen3. The app and benchmark use their documented input behavior, 512-token truncation, attention-masked mean pooling and normalized vectors. E5 prefixes queries with `query: ` and papers with `passage: `; GTE uses plain text. Both model cards state MIT licensing. The current repository pins are:

| Label | Hugging Face repo | Revision | License | Encoding |
|---|---|---|---|---|
| e5-base-v2 | `intfloat/e5-base-v2` | `1c644c92ad3ba1efdad3f1451a637716616a20e8` | MIT | query/passage prefixes, mean pooling, 512 tokens |
| gte-base | `thenlper/gte-base` | `f846641ce4fb595a946b68fe18389798a8da9a64` | MIT | no prefix, mean pooling, 512 tokens |
| bge-base | `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | model card | BGE search instruction, CLS pooling, 512 tokens |
| minilm | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | model card | plain text, mean pooling, 256 tokens |

Official cards: [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2), [GTE-base](https://huggingface.co/thenlper/gte-base). Their task-specific prefixes, pooling examples, and 512-token limits are documented by the model authors. The pinned snapshots link to [E5 commit](https://huggingface.co/intfloat/e5-base-v2/commit/1c644c92ad3ba1efdad3f1451a637716616a20e8) and [GTE commit](https://huggingface.co/thenlper/gte-base/commit/f846641ce4fb595a946b68fe18389798a8da9a64). Existing Phase 3 result files and checkpoint IDs have not been changed.

The Phase 4 runtime preflight found 41.28 GiB free disk, 15.82 GiB installed RAM, 2.22 GiB currently available RAM, and 3.48 GiB available commit. Both selected downloads were attempted sequentially at the pinned revisions and failed before transfer with `WinError 10013` (socket access denied). No model files were added, and no E5/GTE smoke or full validation score exists. BGE-M3/Qwen3 remain cache-only and were not downloaded. The `.venv` already contains the existing MiniLM/BGE-base models.

To stage the two fallback weights later on a network-enabled host, run separately:

```powershell
.\.venv\Scripts\python.exe -m citewise.fetch_model --model e5-base-v2
.\.venv\Scripts\python.exe -m citewise.fetch_model --model gte-base
```

For snapshots copied from another machine, set their local paths before starting the app or benchmark:

```powershell
$env:CITEWISE_MODEL_PATH_E5_BASE_V2 = 'D:\models\e5-base-v2'
$env:CITEWISE_MODEL_PATH_GTE_BASE = 'D:\models\gte-base'
```

The copied directories must be complete SentenceTransformers snapshots from the revisions above. The worker records the pinned revision in its configuration/checkpoint. Avoid changing model or retrieval source within a checkpoint directory.

## Smoke test, then full validation

After both snapshots are available and import/pooling checks pass, first compare them against the existing quality baselines on the same 40 validation questions:

```powershell
New-Item -ItemType Directory -Force artifacts/phase4/smoke-40 | Out-Null
.\.venv\Scripts\python.exe -u -m citewise.research_benchmark --models bm25 minilm bge-base e5-base-v2 gte-base --limit 40 --threads 4 --batch-size 8 --out-dir artifacts/phase4/smoke-40 *> artifacts/phase4/smoke-40/console.log
.\.venv\Scripts\python.exe -m citewise.charts --input artifacts/phase4/smoke-40/benchmark_validation.json --out-dir artifacts/phase4/smoke-40/figures
```

Only after the smoke result and resource use are reviewed, run the complete comparison in its own resumable output directory:

```powershell
New-Item -ItemType Directory -Force artifacts/phase4/full-validation | Out-Null
.\.venv\Scripts\python.exe -u -m citewise.research_benchmark --models bm25 supervised-logistic minilm bge-base e5-base-v2 gte-base --limit 888 --threads 4 --batch-size 8 --out-dir artifacts/phase4/full-validation *> artifacts/phase4/full-validation/console.log
.\.venv\Scripts\python.exe -m citewise.charts --input artifacts/phase4/full-validation/benchmark_validation.json --out-dir artifacts/phase4/full-validation/figures
.\.venv\Scripts\python.exe -m citewise.failure_analysis --out-dir artifacts/phase4/full-validation
.\.venv\Scripts\python.exe -m citewise.research_report --out-dir artifacts/phase4/full-validation --output docs/PHASE4_METHODS_RESULTS.md
```

Create output directories before redirecting the console file. The worker runs one model in each child process, records model load, document encoding, warm query time and memory separately, disables the application disk cache, and resumes paper-level checkpoints. Do not change the held-out test files or pass `--split test`; this harness intentionally reads only validation. The report command above will generate a Phase 4 report from those real results; until a run succeeds, the existing Phase 3 report remains the current measured record.

## Persistent document embeddings

The app's paragraph-vector cache is in `CITEWISE_EMBED_CACHE_DIR` (default `data/cache/embeddings` when running the FastAPI app). Cache keys include model label/revision, paper ID and exact paragraph text, and `.npy` vectors are atomically replaced. Only one loaded SentenceTransformer and one in-process paper matrix are retained at a time. Benchmark workers turn this app cache off so reported encoding times remain comparable.

Precompute the validation corpus on a suitable host, then attach/copy the cache to the deployment volume:

```powershell
.\.venv\Scripts\python.exe -m citewise.prepare_embeddings --model bge-base --split validation --data-dir data/processed --cache-dir data/cache/embeddings --output artifacts/phase4/bge-base-embedding-precompute.json
```

The command does not open `test.jsonl`. Its report records the actual data SHA-256, model revision, paper/paragraph counts, elapsed time, per-paper timing, host packages and peak RSS. Changing paragraph text or model revision invalidates the corresponding cache key.

## UI automation

Pinned Playwright and axe packages are in the root `package.json`; tests exercise real QASPER paper selection, evidence-to-API text parity, bad input, focus styling, horizontal overflow at 390/375 CSS pixels, screenshots and axe WCAG 2.1 A/AA findings. Install and execute them from PowerShell:

```powershell
npm.cmd install
npx.cmd playwright install chromium
npm.cmd run test:e2e
npm.cmd run test:a11y
```

Screenshots, JSON results, and HTML report are written under `artifacts/phase4/`. The attempted `npm.cmd install --ignore-scripts --no-audit --no-fund` made no progress for 60 seconds and was interrupted; no browser test or axe scan has been run in Phase 4. Playwright/axe result artifacts and a generated npm lockfile do not exist yet; generate and retain `package-lock.json` after a successful install. Existing local API tests are not browser evidence. One direct `TestClient` call against actual validation data returned five BM25 passages whose IDs and text matched their source paragraph records; evidence is saved at `artifacts/phase4/ui/api_content_verification.json`. The Docker CLI initially could not read the user Docker config; retrying with workspace-local config showed the Docker Engine named pipe is unavailable. The image was not built.

## Hosting feasibility (checked 2026-10-19)

Hugging Face documents Docker Spaces for FastAPI, with an `sdk: docker` README header and configured `app_port`. The default CPU Basic is 2 vCPU, 16 GB RAM and 50 GB ephemeral disk with no hourly compute charge; creating a new Docker/Gradio compute Space requires a paid plan. CPU Upgrade is documented at $0.03/hour; GPU prices start at $0.40/hour for T4-small. Upgraded machines are billed while running, and unupgraded Spaces may sleep after inactivity. Confirm current account eligibility/terms before creating a Space. [Spaces overview and hardware/pricing](https://huggingface.co/docs/hub/spaces-overview), [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker), [storage behavior](https://huggingface.co/docs/hub/main/spaces-storage).

The Phase 3 BGE-base worker used 1.294 GiB peak RSS on the local 16-CPU host. That supports a CPU-feasibility hypothesis for the 16 GB Space, not a throughput guarantee. Document encoding took about 4,036 seconds for 12,869 paragraphs on that host; precomputation is recommended before public traffic. Bake the pinned model into the Docker image at build time, set `HF_HUB_OFFLINE=1` at runtime, mount prepared validation data read-only, and put the vectors on a persistent Storage Bucket. Space root disk is ephemeral. Keep one Uvicorn worker to avoid duplicating encoder memory.

The deployment skeleton is `Dockerfile` plus `deploy/huggingface/README.md`. It exposes health checks and reads model, dataset, port and cache locations from environment variables. No hosting account or public Space was created; HTTPS, actual Space builds and public end-to-end retrieval are unverified.
