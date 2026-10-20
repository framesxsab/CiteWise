# Phase 5 evaluation freeze and submission status

**Freeze recorded:** 2026-10-19. The four-method QASPER test evaluation was executed once on 2026-10-19 UTC under this freeze. Test outcomes were not used to alter the comparison or model choice.

## Frozen confirmatory comparison

This freeze retains the completed, equal-question four-method comparison. The two additional CPU-oriented encoders are excluded because their pinned weights were unavailable in this execution environment. BGE-M3 and Qwen3 were not initialized. Do not add models, tune parameters, or change the protocol after the final test run begins. The QASPER-trained logistic artifact remains fixed; the training configuration uses balanced logistic regression with `max_iter=1500` and `random_state=42`.

| Method | Role | Frozen checkpoint/artifact |
|---|---|---|
| BM25 | lexical baseline | implementation in the frozen source tree |
| supervised-logistic | QASPER-trained lexical ranker | `artifacts/supervised_ranker.joblib`, SHA-256 `824688c1250c883d2f933fe4a0ee98c5cfd9f373cb1c93e06fb857db2fdc73ec` |
| MiniLM | lightweight pretrained neural retriever | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| BGE-base | selected quality-first retriever | `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |

Dataset: official `allenai/qasper` revision `91bc63e221475d59a6e17e4299fbfc5a919d7c7a`; original paper-level splits and Phase 3 cleaning/mapping rules. Evaluate only mapped textual evidence. Use full original paper paragraphs, `k=5`, macro Recall@5, MRR@5, and binary nDCG@5. Models receive identical papers, queries and gold paragraph IDs. No validation or test fitting is permitted. The logistic model was fitted on train only (`random_state=42`); keep its weak result and negative-sampling diagnosis in the comparison.

MiniLM uses plain text, mean pooling, max sequence length 256. BGE-base uses the documented `Represent this sentence for searching relevant passages: ` query prefix, plain passage text, CLS pooling, max sequence length 512. Both use float32 and L2-normalized vectors. BM25 and the fitted logistic artifact remain unchanged. The final-test evaluation reports quality only; its latency measurements include model/paper cache effects and must not be represented as steady-state service latency.

Validation decision: choose BGE-base for the quality-first demonstration, with MiniLM as the speed-oriented option. On 888 scorable validation questions, measured macro results and 95% paper-cluster CIs were:

| Method | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| BM25 | 44.32% [41.04–47.44] | 34.56% [32.01–37.21] | 32.86% [30.33–35.32] |
| Supervised logistic | 4.93% [3.74–6.27] | 5.20% [4.13–6.38] | 3.91% [3.09–4.86] |
| MiniLM | 46.11% [43.14–48.99] | 36.05% [33.34–38.91] | 34.57% [32.20–36.95] |
| BGE-base | 52.10% [49.20–55.08] | 43.87% [40.99–46.60] | 41.42% [38.98–43.95] |

Paired bootstrap used 2,000 paper-level replicates, seed 42. BGE-base warm P50 was 92.48 ms versus MiniLM 20.07 ms; document encoding took 4,036 s vs 519 s on the measured 16-CPU host. These are validation results, not final test estimates or public-service latency guarantees.

Environment recorded by the validation worker: Windows 11 build 26200, AMD64 16 logical CPUs, Python 3.14.3, 15.82 GiB RAM, CPU-only. Package versions: NumPy 2.5.4, scikit-learn 1.9.1, sentence-transformers 5.7.0, PyTorch 2.14.1, Transformers 5.20.1, Hugging Face Hub 1.33.0, Datasets 4.8.5, FastAPI 0.143.3. The training and bootstrap random seeds are both 42; final ranking/evaluation is deterministic for the recorded software and checkpoint snapshots. Detailed Phase 3 worker configuration is retained in `artifacts/phase3/full-validation-v1/run_evidence.json`.

Current source SHA-256s for the frozen data/training/retrieval/evaluation path (recorded 2026-10-19): `dataset.py` `b1f499d6d06f29e0760396b1affd69b9345bb83852cd9354507825e73f78f644`; `training.py` `103c582e6462d63ccd100153fd024e28d549ef8a51e4d232e1b03dff0657c0bd`; `retrieval.py` `f942d37b791fbabc74104b77f566e034cec385f0a688fad09bc15d46c2e1bd39`; `evaluation.py` `983591430454575e93f9874c8d41f19693a3ec5c38a43d0997df278879b96a41`; `research_benchmark.py` `7d0aca1947835ed810ea958ff86fcaa8714549e58923cbb704d00d100d56c64e`; `benchmark.py` `ec5284c4b0a89621bc7fab087cb170c39f1d8e2b73e8cc04860b789e5fa5a57e`.

## Single held-out run (executed once; do not repeat)

The exact frozen command below was run once. It read the official test JSONL and wrote to a separate final-test directory. **Do not run it again**; use its saved JSON/CSV outputs for plots, bootstrap intervals and report assembly.

```powershell
New-Item -ItemType Directory -Force artifacts/phase5/final-test | Out-Null
.\.venv\Scripts\python.exe -m citewise.evaluation --split test --models bm25 supervised-logistic minilm bge-base --out-dir artifacts/phase5/final-test
.\.venv\Scripts\python.exe -m citewise.charts --input artifacts/phase5/final-test/evaluation_test.json --out-dir artifacts/phase5/final-test/figures
```

Completed results: 1,309 scorable questions in 408 paper clusters for each method; 63 unmatched/no-text-evidence and 79 unanswerable questions were excluded consistently. Macro means and 95% paper-cluster CIs:

| Method | Recall@5 | MRR@5 | nDCG@5 | Observed P50/P95 ms |
|---|---:|---:|---:|---:|
| BM25 | 43.77% [41.38–46.24] | 37.92% [35.63–40.19] | 34.57% [32.61–36.58] | 2.88 / 6.13 |
| Supervised logistic | 5.50% [4.53–6.60] | 6.66% [5.54–7.91] | 4.87% [4.03–5.85] | 7.60 / 14.71 |
| MiniLM | 47.36% [44.92–49.87] | 43.70% [41.34–46.07] | 38.77% [36.66–40.90] | 25.98 / 1,684.73 |
| BGE-base | 52.35% [49.84–54.86] | 49.97% [47.45–52.30] | 44.32% [42.18–46.47] | 110.37 / 12,887.97 |

Intervals were calculated from saved per-query predictions only (2,000 paper-cluster bootstrap replicates, seed 42); the raw test JSONL was not reloaded. Paired BGE-base minus MiniLM improvements were +4.99 pp Recall@5 [2.88, 7.08], +6.27 pp MRR@5 [4.44, 8.17], and +5.55 pp nDCG@5 [3.95, 7.17]. BGE-base minus BM25 improvements were +8.57 pp [5.93, 11.12], +12.05 pp [9.48, 14.81], and +9.75 pp [7.62, 12.05]. These are descriptive test comparisons; they did not alter the frozen selection. No multiple-comparison correction was applied.

Peak RSS for the combined evaluation process was 1.325 GiB. The run took 75 minutes 21 seconds on the CPU-only host. The legacy `evaluation.py` timer includes lazy model loading and first-time per-paper encoding in the request latency. Thus its P95 values are observed traversal latencies and are not steady-state service latency; model loading, per-model encoding and warm retrieval were not separated in this test run.

Do not use the resulting test metrics to change model choice, training, features, preprocessing or metrics. The optional E5/GTE fallback comparison was excluded from this freeze; its artifacts and weights remain unavailable/unmeasured in this environment. Any future experiment requires a new model-selection protocol and must not be described as the test evaluation run again.

## Browser evidence

The Playwright run used locally cached exact versions Playwright Test 1.63.0 and axe-core Playwright 4.13.0, with Chromium 153.0.8010.12 (revision 1243). The project lockfile was installed offline. Against the real local FastAPI app and QASPER validation records, `npm run test:e2e` passed six tests: BM25 evidence/API parity, cached BGE-base neural evidence/API parity, loading feedback, invalid question/API requests, keyboard focus, desktop no-overflow/evidence, mobile no-overflow with evidence at 390 and 375 CSS pixels, and axe scan. Standalone `npm run test:a11y` passed; axe reported zero WCAG 2.1 A/AA violations. These automated browser checks are not human usability testing or assistive-technology testing.

Genuine browser artifacts: screenshots at `artifacts/phase5/ui/final/`, Playwright HTML/JSON report at `artifacts/phase5/final-browser-report/`, axe output at `artifacts/phase5/a11y/final/axe-results.json`. Reproduce in two PowerShell terminals:

```powershell
# Terminal 1: run the local app
$env:CITEWISE_DEVICE = 'cpu'
$env:CITEWISE_DEFAULT_MODEL = 'bm25'
$env:CITEWISE_EMBED_CACHE_DIR = 'artifacts/phase5/browser-embeddings'
.\.venv\Scripts\python.exe -m uvicorn citewise.app:app --host 127.0.0.1 --port 8765 --workers 1

# Terminal 2: run the browser checks
$env:CITEWISE_BASE_URL = 'http://127.0.0.1:8765'
npm ci
npm run test:e2e
npm run test:a11y
```

Browser base URL was local; no public hostname or HTTPS endpoint was tested.

## Environment and deployment boundary

Within this Codex execution environment, `Test-NetConnection huggingface.co -Port 443` and `Test-NetConnection registry.npmjs.org -Port 443` both returned `TcpTestSucceeded: False`. The prior E5/GTE downloads had failed with `WinError 10013`; neither checkpoint is cached. This establishes an outbound-connectivity blocker in this environment, but cannot determine whether the user's ordinary Windows PowerShell has the same restriction. No firewall or security setting was changed. To distinguish host connectivity from the Codex execution boundary, run these from regular PowerShell outside Codex:

```powershell
Test-NetConnection huggingface.co -Port 443
Test-NetConnection registry.npmjs.org -Port 443
docker version
```

Docker CLI 29.5.3 is installed, but `docker version` cannot connect to `npipe:////./pipe/docker_engine`; the Docker Engine is unavailable to this process. No image build or public deployment occurred. Hugging Face hosting-account access was not established. Public HTTPS, cold-start behavior on a host, mounted-volume permissions and cross-platform deployment remain unverified.

The four measured methods fulfill a four-method comparison; only two are neural retrievers. Do not claim four state-of-the-art neural models. Phase 3 result artifacts remain unchanged, and all Phase 5 results and screenshots are separate.
