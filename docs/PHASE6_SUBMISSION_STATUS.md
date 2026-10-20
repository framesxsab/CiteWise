# Phase 6 submission status

Updated 2026-10-20. The final held-out evaluation was run exactly once on 2026-10-19 under `docs/PHASE5_FREEZE.md`. Do not rerun it.

## Faculty assessment checklist

| Area | Evidence | Status and limitation |
|---|---|---|
| Dataset, cleaning and official split integrity | QASPER revision and split audit in the report; original paper-level splits retained | Complete for the measured study |
| Model training and comparison | Train-fitted logistic model; BM25, logistic, MiniLM and BGE-base validation and one-time test results | Four retrieval methods measured, but only two neural models; four state-of-the-art neural models are not established |
| Metrics and analysis | Recall@5, MRR@5, nDCG@5, 95% paper-cluster intervals, paired comparison chart and failure analysis | Complete for the four measured methods; test latency includes cold/cache effects and is not warm-service latency |
| User interface | Local Playwright tests against real validation papers, evidence/API parity, keyboard focus, loading/error states, desktop and 375/390px layout | Automated local checks passed; no manual assistive-technology review |
| Deployment | Dockerfile and deployment configuration are present | Docker Engine unavailable; no image build, hosting account, public URL or HTTPS user flow verified |
| Report and literature | Exactly ten-page report with seven cited works and measured figures | Complete as a local artifact; five recent retrieval/RAG papers (2024-2025) were checked against primary publisher/preprint pages |
| GitHub publication readiness | Local Git repository, ignore rules, Python and npm locks, setup/testing docs | Not published; root source-code `LICENSE` is absent and must be selected by the project owner |

## Frozen held-out results

All methods used the same 1,309 scorable test questions from 408 paper clusters. 63 questions without mapped text evidence and 79 unanswerable questions were excluded. Intervals use 2,000 paper-cluster bootstrap replicates with seed 42.

| Method | Recall@5 (95% CI) | MRR@5 (95% CI) | nDCG@5 (95% CI) | Observed P50/P95 ms |
|---|---:|---:|---:|---:|
| BM25 | 43.77% [41.38, 46.24] | 37.92% [35.63, 40.19] | 34.57% [32.61, 36.58] | 2.88 / 6.13 |
| Supervised logistic | 5.50% [4.53, 6.60] | 6.66% [5.54, 7.91] | 4.87% [4.03, 5.85] | 7.60 / 14.71 |
| MiniLM | 47.36% [44.92, 49.87] | 43.70% [41.34, 46.07] | 38.77% [36.66, 40.90] | 25.98 / 1,684.73 |
| BGE-base | 52.35% [49.84, 54.86] | 49.97% [47.45, 52.30] | 44.32% [42.18, 46.47] | 110.37 / 12,887.97 |

The combined evaluation process used 1.325 GiB peak RSS and ran for 75 minutes 21 seconds. The reported latency distribution includes lazy model setup and first-time paper encoding; it must not be labeled steady-state inference. No test result was used to change model selection.

## Verification evidence

- Python: 28 passed, one existing Starlette/httpx deprecation warning (full suite after the evaluation post-processing additions).
- Browser: 6 passed; Playwright 1.63.0 with Chromium 153.0.8010.12. Axe: one scan passed with zero WCAG 2.1 A/AA violations.
- Phase 6 rerun: the 390px evidence-flow browser case passed (1/1) and wrote the mobile evidence screenshot.
- Report build: `python submission/build_report.py` produced a ten-page PDF; the generator checked each fixed page for overflow and verified ten PDF page objects. Page renders are retained under `submission/qa/`.
- Docker/public host: not verified. Docker CLI 29.5.3 cannot connect to its engine pipe. Host-side connectivity/account access is unknown.

## Reproduction and next actions

Regenerate the report after editing its Markdown source:

```powershell
.\.venv\Scripts\python.exe submission\build_report.py
```

Run local checks without touching the held-out split:

```powershell
.\.venv\Scripts\python.exe -m pytest
$env:CITEWISE_BASE_URL = 'http://127.0.0.1:8765'
npm.cmd run test:e2e
npm.cmd run test:a11y
```

From regular Windows PowerShell, check whether host networking and Docker are available before attempting optional model staging or a build:

```powershell
Test-NetConnection huggingface.co -Port 443
Test-NetConnection registry.npmjs.org -Port 443
docker version
```

Before public repository distribution, the owner must choose a project-code license. Before deployment claims, build and exercise the image on a working Docker Engine, provision authorized hosting, and verify an HTTPS search flow. Neither action has been performed. The held-out evaluation is complete and must not be repeated.
