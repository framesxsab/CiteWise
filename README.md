# CiteWise: Evidence-Grounded Scientific Retrieval

**Mini-project scope:** QASPER research-paper evidence retrieval, a fair comparison of four measured methods (BM25, a train-fitted logistic ranker, MiniLM and BGE-base), and a mobile-responsive live demo. Only MiniLM and BGE-base are neural retrievers. E5/GTE, BGE-M3 and Qwen3 have no CiteWise measurements in this submission state.

**No dummy project data.** The user-facing demo reads real QASPER papers only after `prepare` runs. Unit tests deliberately use tiny fixture strings to test the algorithms; they are **not** training, evaluation or demo data. No benchmark results are fabricated or shipped.

## Dataset provenance
- Dataset: `allenai/qasper` (CC BY 4.0); 1,585 research papers and 5,049 human-written questions.
- Use the **official paper-level splits**, not random paragraph splitting: 888 train, 281 validation, 416 test papers.
- Training: fit a transparent supervised logistic evidence ranker on real train papers. Validation: choose models and parameters. The official test split was held out through selection and evaluated once under the Phase 5 freeze; see the Phase 6 report.
- Evidence annotations are normalized and matched against original paper paragraphs. Unanswerable annotations and annotations referring to figures/tables cannot silently become ground-truth text; audit counts are reported in `data_audit.json`.
- The scope is **within-paper paragraph retrieval**, not open-web scientific fact-checking or complete end-to-end answer generation.

## Quick start (Windows PowerShell or Linux/macOS shell)

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e '.[models,charts,dev]'
python -m citewise.prepare
python -m citewise.training
python -m citewise.ranker_diagnostics --limit 40 --candidate-k 20 --out-dir artifacts/phase2/ranker-top20
python -m citewise.benchmark --models bm25 minilm bge-base --limit 40 --out-dir artifacts/phase2/benchmark
python -m citewise.charts --input artifacts/phase2/benchmark/benchmark_validation.json --out-dir artifacts/phase2/benchmark/figures
uvicorn citewise.app:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000` on your computer. The UI uses real papers, input questions and actual retrieval scores.

For the recorded Windows validation environment, install the pinned packages and project source without allowing dependency resolution to change the lock:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-win-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

The Phase 2 benchmark defaults to the three models already available for local work. It runs offline, separates model loading and paper embedding from warm query ranking, writes per-query and paired paper-bootstrap evidence, and never reads the test split. The original `artifacts/evaluation_validation.json` and `artifacts/bge40/evaluation_validation.json` remain the earlier baseline records.

For model-specific commands, the train-only ranker diagnosis, the hosted GPU workflow, and the final test-set freeze gate, see [Phase 2 evaluation](docs/PHASE2_EVALUATION.md). BGE-M3 and Qwen3 remain optional and are cache-only in the application; they are not downloaded by the benchmark.

## Phase 3 full validation

The resumable Phase 3 command evaluates all 888 scorable validation questions with the currently cached models and the unchanged supervised baseline. Each model runs in a separate process; checkpoints are saved after each paper. It does not read the test split.

```powershell
New-Item -ItemType Directory -Force artifacts/phase3/full-validation-v1 | Out-Null
.\.venv\Scripts\python.exe -u -m citewise.research_benchmark --models bm25 supervised-logistic minilm bge-base --limit 888 --threads 4 --batch-size 16 --out-dir artifacts/phase3/full-validation-v1 *> artifacts/phase3/full-validation-v1/console.log
.\.venv\Scripts\python.exe -m citewise.charts --input artifacts/phase3/full-validation-v1/benchmark_validation.json --out-dir artifacts/phase3/full-validation-v1/figures
.\.venv\Scripts\python.exe -m citewise.failure_analysis --out-dir artifacts/phase3/full-validation-v1
.\.venv\Scripts\python.exe -m citewise.research_report --out-dir artifacts/phase3/full-validation-v1 --output docs/REPORT_METHODS_RESULTS.md
```

That full run completed and its evidence is preserved. Phase 4 subsequently changed the retrieval source hash, so the original `full-validation-v1` checkpoints must not be resumed under the changed source. A new run uses a new output directory. Checkpointing requires matching hashes for data, query plan, source, model snapshots and settings. `docs/PHASE3_PROTOCOL.md` records the original protocol and the `e5-base-v2`/`gte-base` labels do not imply metrics until a real validation run succeeds.

For a basic-only CPU environment, install `pip install -e '.[dev]'` and use `--models bm25` (the interface and tests work without downloading neural weights). **BM25 alone does not fulfill the final four-model requirement.**

## Evaluation discipline

- Each model is evaluated on **the same scored questions**, scoped to its paper's original full-text paragraphs.
- Metrics: macro Recall@5, MRR@5, nDCG@5, query-time P50/P95 latency. Report dataset sizes, exact number of scorable questions and excluded cases. For multiple annotated gold paragraphs, Recall@5 is the fraction of relevant gold passages retrieved.
- Only score questions whose gold textual evidence maps to the paper paragraphs. Report coverage and unanswerable cases separately, rather than hiding them as errors or marking them correct.
- Preserve identical retrieval scope, `top_k`, normalization and CPU/GPU settings. Record model setup and version. Repeat timing runs if performance analysis is central.
- **Do not tune on the test set.** Fix model choice, preprocessing, `k`, and any supervised parameters on train/validation first.
- Neural models are used as *pretrained encoders*, not falsely described as individually fine-tuned on QASPER. The supervised logistic ranker is genuinely fitted to QASPER training evidence.
- User questions without gold annotations are for the demonstration, **not** part of the measured test score.

## Files

- `src/citewise/dataset.py`: QASPER normalization, paper-level splits, evidence coverage audit
- `src/citewise/retrieval.py`: BM25 and six configured neural sentence-embedding models (heavy models remain cache-only)
- `src/citewise/training.py`: learned logistic ranker trained on gold evidence from the train split only
- `src/citewise/evaluation.py`: reproducible held-out evidence metrics and per-query CSV
- `src/citewise/ranker_diagnostics.py`: training-feature diagnostics and bounded train/validation-only ranker experiments
- `src/citewise/benchmark.py`: offline phase-separated benchmark, paired query exports, failures and paper-group bootstrap intervals
- `src/citewise/research_benchmark.py`: isolated model workers, per-paper checksums, resume support, token audit and full-validation evidence
- `src/citewise/fetch_model.py`, `src/citewise/prepare_embeddings.py`: explicit pinned weight staging and validation/train-only persistent document-vector precomputation
- `src/citewise/failure_analysis.py`: BM25/neural disagreement cases, long-input flags and excluded-QASPER-question export
- `src/citewise/accessibility_audit.py`: source structure and declared-palette accessibility checks (browser rules remain separate)
- `src/citewise/research_report.py`: report tables generated from completed result files
- `src/citewise/charts.py`: figure generation from measured results only
- `src/citewise/app.py`, `src/citewise/index.html`: FastAPI and responsive browser UI
- `tests/`: parser, evidence mapping, retrieval and metric tests
- `docs/REPORT_BLUEPRINT.md`: 10-page deliverable structure
- `docs/PHASE2_EVALUATION.md`: Phase 2 experiment protocol and hosted GPU workflow
- `docs/PHASE3_PROTOCOL.md`: frozen full-validation protocol, checkpointing and model availability boundary
- `docs/REPORT_METHODS_RESULTS.md`: generated methodology/results material when Phase 3 validation finishes
- `docs/LITERATURE_COMPARISON.md`: source-backed six-paper review and critical comparison
- `docs/PHASE5_FREEZE.md`: frozen four-method comparison, test-once command, and submission evidence boundaries
- `docs/LICENSING.md`: dataset/model licenses and the undeclared project-code license
- `requirements-win-lock.txt`: exact Windows Python package environment recorded for validation

## Phase 4 model staging and app cache

The full-validation record remains in `artifacts/phase3/full-validation-v1/` and is not overwritten by Phase 4. E5-base-v2 and GTE-base are configured with official asymmetric/symmetric input handling, 512-token limits, mean pooling, L2 normalization and pinned MIT-licensed repository commits. They are older fallback encoders, not substitutes for the BGE-M3/Qwen3 target comparison. The current machine's download attempt failed with `WinError 10013`; neither model has validation metrics yet. `CITEWISE_MODEL_PATH_E5_BASE_V2` and `CITEWISE_MODEL_PATH_GTE_BASE` allow a user to point to separately staged local snapshots. See `docs/PHASE4_READINESS.md`.

The UI's provisional default is BGE-base (`CITEWISE_DEFAULT_MODEL`), and inference is CPU-only in the prepared image. `CITEWISE_EMBED_CACHE_DIR` can point to a persistent vector cache. `python -m citewise.prepare_embeddings --model bge-base --split validation` writes normalized paragraph vectors and a hardware/timing report; this command only reads the selected train or validation file. Benchmark workers explicitly disable this application cache so document-encoding timings remain cold and comparable.

## Deployment

The Docker image stages one selected, pinned model during image build and disables Hub access at runtime. It sets a one-worker Uvicorn process, configurable port, environment-based dataset/cache/model settings and an `/api/health` container check. Use `deploy/huggingface/README.md` for the Hugging Face Docker Space template and `docs/DEPLOYMENT_STATUS.md` for current verification and blockers. The demo requires prepared validation data mounted at `/data/processed`; persistent vectors belong on a read-write volume at `/data/cache`. Model weights are built into the image. A model build needs network access at image-build time, but serving requests does not download weights.

The repository includes Playwright desktop/mobile, real BM25 and BGE-base evidence/API parity, invalid-input/loading-state, keyboard-focus and axe tests. `package-lock.json` pins the browser tools. Start the local API on port 8765, then run `npm ci`, `npx playwright install chromium`, `npm run test:e2e`, and `npm run test:a11y`. Current local browser evidence is in `artifacts/phase5/`; see `docs/PHASE5_FREEZE.md` and `docs/PHASE6_SUBMISSION_STATUS.md`. The public deployment is still unverified.

## Phase 5 verification

The frozen comparison uses the four methods already measured on validation: BM25, the train-fitted logistic ranker, MiniLM and BGE-base. Only MiniLM and BGE-base are neural models. BGE-base was selected on validation for quality, with MiniLM as the faster alternative. E5/GTE, BGE-M3 and Qwen3 have no CiteWise scores. The held-out test was evaluated once after the freeze and must not be rerun; results are in `artifacts/phase5/final-test/` and the 10-page report is in `submission/`. The local browser suite passed 6/6 tests and axe reported zero WCAG 2.1 A/AA violations; reports and screenshots are under `artifacts/phase5/`. A public deployment remains unverified because Docker Engine and authorized hosting were unavailable.

## Licensing

QASPER is CC BY 4.0; MiniLM is Apache-2.0; BGE-base is MIT. A root code license has not been assigned. See `docs/LICENSING.md` before public repository release.

## Phase 6 submission status

The final held-out run has been performed once; its metrics, paper-cluster confidence intervals and caveats are in the exactly ten-page `submission/CiteWise_Final_Report.pdf` and Markdown source in `docs/FINAL_REPORT.md`. Regenerate the PDF with `python submission/build_report.py`; it uses the locked `markdown-it-py` and Playwright packages. The raw per-query files remain local under `artifacts/phase5/final-test/` and are ignored by Git to avoid publishing copied QASPER passages. Reproduce the measured Windows environment with `pip install -r requirements-win-lock.txt` followed by `pip install --no-deps -e .`. Browser dependencies use `npm ci`.

Code license is not assigned yet; see `docs/LICENSING.md`. Do not imply a public GitHub release or deployment. The project directory is prepared locally for Git, but no commit or push was made.

## Caveats

The retrieval output is a set of cited evidence *candidates*, not a trustworthy generated answer. The project should not present paraphrased LLM output as independently verified scientific truth. Later expansions can add an evidence-grounded answer generator, uncertainty-based abstention, PDF upload with access controls, and AutoSD device adapters.
