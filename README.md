# CiteWise: Evidence-Grounded Scientific Retrieval

**Current study:** CiteWise retrieves evidence paragraphs from real QASPER research papers. The Phase 9 validation comparison measures four neural retrievers (MiniLM-L6-v2, BGE-base-en-v1.5, E5-base-v2 and GTE-base) plus BM25 and a train-fitted logistic baseline on 888 scorable questions from 276 papers. The older once-only held-out evaluation remains the original four-method result; E5 and GTE have no test scores.

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

The UI's validation-selected quality-first default is E5-base-v2 (`CITEWISE_DEFAULT_MODEL`); MiniLM remains the speed-oriented alternative. Inference is CPU-only in the prepared image. `CITEWISE_EMBED_CACHE_DIR` can point to a persistent vector cache. `python -m citewise.prepare_embeddings --model e5-base-v2 --split validation` writes normalized paragraph vectors and a hardware/timing report; this command only reads the selected train or validation file. Benchmark workers explicitly disable this application cache so document-encoding timings remain cold and comparable.

## Deployment

The Docker image stages one selected, pinned model during image build and disables Hub access at runtime. It sets a one-worker Uvicorn process, configurable port, environment-based dataset/cache/model settings and an `/api/health` container check. Use `deploy/huggingface/README.md` for the Hugging Face Docker Space template and `docs/DEPLOYMENT_STATUS.md` for current verification and blockers. The demo requires prepared validation data mounted at `/data/processed`; persistent vectors belong on a read-write volume at `/data/cache`. Model weights are built into the image. A model build needs network access at image-build time, but serving requests does not download weights.

The repository includes Playwright desktop/mobile, real BM25 and BGE-base evidence/API parity, invalid-input/loading-state, keyboard-focus and axe tests. `package-lock.json` pins the browser tools. Start the local API on port 8765, then run `npm ci`, `npx playwright install chromium`, `npm run test:e2e`, and `npm run test:a11y`. Current local browser evidence is in `artifacts/phase5/`; see `docs/PHASE5_FREEZE.md` and `docs/PHASE6_SUBMISSION_STATUS.md`. The public deployment is still unverified.

## Phase 5 verification

The frozen comparison uses the four methods already measured on validation: BM25, the train-fitted logistic ranker, MiniLM and BGE-base. Only MiniLM and BGE-base are neural models. BGE-base was selected on validation for quality, with MiniLM as the faster alternative. E5/GTE, BGE-M3 and Qwen3 have no CiteWise scores. The held-out test was evaluated once after the freeze and must not be rerun; results are in `artifacts/phase5/final-test/` and the 10-page report is in `submission/`. The local browser suite passed 6/6 tests and axe reported zero WCAG 2.1 A/AA violations; reports and screenshots are under `artifacts/phase5/`. A public deployment remains unverified because Docker Engine and authorized hosting were unavailable.

## Licensing

QASPER is CC BY 4.0; MiniLM is Apache-2.0; BGE-base is MIT. A root code license has not been assigned. See `docs/LICENSING.md` before public repository release.

## Phase 6 submission status

The final held-out run has been performed once; its metrics, paper-cluster confidence intervals and caveats are in the exactly ten-page `submission/CiteWise_Final_Report.pdf` and Markdown source in `docs/FINAL_REPORT.md`. Regenerate the PDF with `python submission/build_report.py`; it uses the locked `markdown-it-py` and Playwright packages. The raw per-query files remain local under `artifacts/phase5/final-test/` and are ignored by Git to avoid publishing copied QASPER passages. Reproduce the measured Windows environment with `pip install -r requirements-win-lock.txt` followed by `pip install --no-deps -e .`. Browser dependencies use `npm ci`.

The repository is published at `https://github.com/framesxsab/CiteWise` (current published baseline `e700a81`). Phase 7 changes in this working tree are not yet committed or pushed. A root source-code license is still unassigned; see `docs/LICENSING.md`. GitHub publication does not imply a hosted application deployment.

## Phase 7 faculty integration (historical snapshot)

The existing FastAPI UI now searches the validation split only, disables neural retrievers whose pinned weights are not staged, and displays the saved full-validation comparison and dataset audit. Recall@1 and its paper-cluster interval are derived from the saved Phase 3 validation rankings; no model inference was rerun to produce them. The interface shows BM25 and the logistic ranker as baselines, MiniLM/BGE-base as measured neural retrievers, and E5/GTE as unmeasured. It does not display held-out test metrics. Rebuild its aggregate summary from the preserved validation evidence with:

```powershell
.\.venv\Scripts\python.exe -m citewise.dashboard_data
```

At the Phase 7 snapshot, Python had `33 passed`, Playwright `7 passed`, and axe reported `0 violations`. E5/GTE had not yet been scored. The Phase 8 validation below supersedes that model-status snapshot; public deployment remains unverified.

## Phase 8 validation and checkpoint recovery

The resumable, CPU-only Phase 8 benchmark evaluated BM25, supervised logistic, MiniLM-L6-v2, BGE-base-en-v1.5, E5-base-v2 and GTE-base on the same **888 scorable validation questions from 276 paper clusters** (the validation split contains 1,005 total questions; 57 lacked mapped textual evidence and 60 were unanswerable). Results and per-query evidence are in `artifacts/phase8/full-validation/`; `src/citewise/research_summary.json` is the application dashboard source. E5 leads the validation point estimates: Recall@1 21.66%, Recall@5 54.67%, MRR@5 46.09%, nDCG@5 43.40%. Its paired advantage over BGE-base is clearest for Recall@5; paired MRR and nDCG intervals include zero. MiniLM has the lowest warm P50 among the neural models (18.81 ms). These are local CPU retrieval measurements, not hosted-service performance.

The BM25 run resumed from 157 verified paper checkpoints after a Windows manifest-replacement failure. The checkpoint writer now uses closed, flushed same-directory temporary files and bounded retry for transient Windows sharing errors; recovery validates checksums, run configuration, paper identity and query coverage. Existing checkpoint files were retained. The full aggregate records the narrowly scoped checkpoint-I/O source compatibility migration.

Run the complete Python verification suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The current Phase 8 report is `submission/CiteWise_Final_Report_Phase8.pdf` (10 pages); the earlier Phase 6 PDF remains preserved. Phase 8 does not change the frozen four-method held-out results and does not expose them in the application. Playwright and axe evidence for the Phase 8 application is under `artifacts/phase8/`; these automated local checks do not establish public deployment or manual assistive-technology verification.

## Phase 9 final analysis and faculty demonstration

The verified result summary is built only from the saved Phase 8 validation aggregate, per-query CSV and official prepared validation data. It checks question coverage, source-passage IDs, per-query metrics, config fingerprints, paper-cluster intervals and saved paired comparisons before writing the dashboard bundle and `artifacts/phase9/model_comparison.csv`. Recall@1 is derived from the first saved ranked evidence ID. No model inference is run by these summary or chart commands.

E5-base-v2 leads the validation point estimates (Recall@1 21.66%, Recall@5 54.67%, MRR@5 46.09%, nDCG@5 43.40%). Its paired Recall@5 interval versus BGE-base narrowly excludes zero; paired Recall@1, MRR and nDCG intervals include zero. MiniLM is the fastest neural model (warm P50/P95 18.81/26.22 ms) and uses 0.70 GiB peak RSS. These local CPU timings are not hosted-service measurements. The app defaults to E5 when its pinned weights are staged, otherwise falls back to BM25; MiniLM remains selectable as the speed-oriented model.

Regenerate verified UI metrics and figures without rerunning inference:

```powershell
.\.venv\Scripts\python.exe -m citewise.dashboard_data --benchmark artifacts/phase8/full-validation/benchmark_validation.json --predictions artifacts/phase8/full-validation/per_query_validation.csv --data-audit data/processed/data_audit.json --validation-data data/processed/validation.jsonl --out src/citewise/research_summary.json --comparison-csv artifacts/phase9/model_comparison.csv
.\.venv\Scripts\python.exe -m citewise.charts --input artifacts/phase8/full-validation/benchmark_validation.json --predictions artifacts/phase8/full-validation/per_query_validation.csv --summary src/citewise/research_summary.json --out-dir artifacts/phase9/figures
```

Phase 9 verification: Python `44 passed` (one Starlette/httpx deprecation warning), Playwright `10 passed`, standalone axe `1 passed` with `0 violations`, and wheel packaging passed with the updated UI and summary included. Browser screenshots and reports are under `artifacts/phase9/`; the ten report page renders are under `submission/qa-phase9/`. Manual assistive-technology testing was not performed.

Build the versioned report without replacing the Phase 6 or Phase 8 PDFs:

```powershell
.\.venv\Scripts\python.exe submission\build_report.py --source docs/FINAL_REPORT_PHASE9.md --name CiteWise_Final_Report_Phase9 --qa-dir qa-phase9
```

The report source and PDF are `docs/FINAL_REPORT_PHASE9.md` and `submission/CiteWise_Final_Report_Phase9.pdf`; the Phase 8 report remains preserved. `docs/VIVA_PHASE9.md` contains concise viva explanations and a local-demo walkthrough. The default model and Docker build configuration use E5 when staged, but no Docker image or public HTTPS deployment is verified. The application falls back to BM25 when E5 weights or validation data are absent. Manual screen-reader testing also remains outstanding.

## Phase 10 localhost faculty demo

Use [the Phase 10 localhost demo guide](docs/LOCAL_DEMO_PHASE10.md) for the Windows startup command, two genuine QASPER validation questions verified through all six search methods, a model/cache warm-up preflight and a 5-7-minute walkthrough. The application stays on localhost and validation-only. Preflight responses are operational checks, not benchmark measurements.

Phase 10 local check: Uvicorn health returned ready on port 8765; both demo questions returned five source-matching passages for all six methods, with annotated evidence in each top five. Python passed `44` tests; Playwright passed `11`; standalone axe found `0` violations. The Phase 9 report remains exactly ten pages.

Phase 10 verification used Uvicorn at `127.0.0.1:8765`; health was ready, all six methods were available, and test-split requests returned 422. The API demo verified five source-matching results for each of the two examples, with annotated gold evidence present in every method's top five. Python passed `44/44`; Playwright passed `11/11`; standalone axe reported `0` violations; wheel packaging passed. The Phase 9 ten-page report remains unchanged.

Phase 10 verification: Python `44 passed`; Playwright `11 passed` using one worker; standalone axe `0 violations`; wheel contains `citewise/index.html` and the six-model summary. Example-search screenshots and the executed warm-up log are in `artifacts/phase10/`. The desktop/mobile checks use local Chromium; public hosting, extra browser engines/physical devices and manual assistive-technology testing are not claimed.

## Caveats

The retrieval output is a set of cited evidence *candidates*, not a trustworthy generated answer. The project should not present paraphrased LLM output as independently verified scientific truth. Later expansions can add an evidence-grounded answer generator, uncertainty-based abstention, PDF upload with access controls, and AutoSD device adapters.
