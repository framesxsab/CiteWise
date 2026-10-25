# CiteWise Phase 10: localhost faculty demo

Verified on the Windows project environment using the real FastAPI search endpoint and prepared QASPER validation records. The app was run locally at `http://127.0.0.1:8765`; `/api/health` returned `ready`, `/api/config` found the validation file and all six methods, and the E5 default was available. The full validation benchmark was not rerun.

## One PowerShell startup command

From the repository root, run this single PowerShell command:

```powershell
$env:CITEWISE_DATA_DIR='data/processed'; $env:CITEWISE_EMBED_CACHE_DIR='data/cache/embeddings'; $env:CITEWISE_DEFAULT_MODEL='e5-base-v2'; $env:CITEWISE_DEVICE='cpu'; $env:CITEWISE_BATCH_SIZE='8'; .\.venv\Scripts\python.exe -m uvicorn citewise.app:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. The app validates model availability against both the staged weights/ranker and `data/processed/validation.jsonl`. If E5 is unavailable, it selects BM25. The interactive API accepts only the validation split.

The app caches one neural encoder at a time to limit CPU memory and stores normalized paragraph vectors under `data/cache/embeddings`. A model switch can reload weights; a cached paper vector avoids re-encoding that paper. A preflight warm-up can pay the first model-runtime import and write the selected paper vectors before the viva. Do not present warm-up/API `latency_ms` as the Phase 9 warm-latency benchmark: first-call times depend on initialization and cache state.

One un-prewarmed localhost pass returned 79,516 ms for MiniLM's first search and 21,574 ms for the logistic model's first search; those API timings include lazy setup/encoding and are not steady-state latency claims. Run the preflight before the viva rather than showing first-use initialization during the timed demonstration.

## Genuine validation questions and retrieved evidence

Both questions are annotated in the official validation split. The API returned five unique passages for each method, every returned ID/text matched its prepared source paper, and each of BM25, supervised logistic, MiniLM, BGE-base, E5-base-v2 and GTE-base returned at least one gold passage in its top five.

Paper: **Creative GANs for generating poems, lyrics, and metaphors** (`1909.09534`; 7 paragraphs).

| Annotated question | Question ID | Gold evidence IDs | Verified gold rank in top five: BM25 / logistic / MiniLM / BGE / E5 / GTE |
|---|---|---|---|
| Which datasets are used? | `fc77d70c305fa80447b191248aba93da63ac3704` | `1909.09534:s1:p1`, `1909.09534:s1:p2` | 2,3 / 4,5 / 2 / 2,3 / 2,3 / 2,4 |
| What objective function is used in the GAN? | `6411622cc8b2fbedbfa468859d453596d3bd2f03` | `1909.09534:s1:p2` | 3 / 4 / 2 / 3 / 4 / 3 |

The first question's mapped passages describe datasets of English poems, metaphor sentences and song lyrics. The second passage describes Gutenberg pretraining, a language-modeling objective for fine-tuning, and an adversarial discriminator objective. Show the passages and citations as evidence candidates; CiteWise does not generate or verify answers.

### Optional warm-up and verification command

Run this from another PowerShell after starting the app. It performs bounded, real searches for both demonstration questions across all six methods, verifies that each returns five passages and at least one annotated gold ID, then records the warm-up log. This is operational preflight only; it does not calculate or update evaluation metrics.

```powershell
$base = 'http://127.0.0.1:8765'
$paperId = '1909.09534'
$models = @('bm25', 'supervised-logistic', 'minilm', 'bge-base', 'gte-base', 'e5-base-v2')
$examples = @(
    [pscustomobject]@{ Question = 'Which datasets are used?'; Gold = @('1909.09534:s1:p1', '1909.09534:s1:p2') },
    [pscustomobject]@{ Question = 'What objective function is used in the GAN?'; Gold = @('1909.09534:s1:p2') }
)

$lines = foreach ($model in $models) {
    foreach ($example in $examples) {
        $body = @{
            split = 'validation'
            paper_id = $paperId
            question = $example.Question
            model = $model
            top_k = 5
        } | ConvertTo-Json
        $result = Invoke-RestMethod -Uri "$base/api/search" -Method Post `
            -ContentType 'application/json' -Body $body -TimeoutSec 180
        $ids = @($result.results | ForEach-Object id)
        if ($ids.Count -ne 5) { throw "$model returned $($ids.Count) passages" }
        $hits = @($ids | Where-Object { $_ -in $example.Gold })
        if ($hits.Count -eq 0) { throw "$model returned no annotated evidence for: $($example.Question)" }
        "{0} | {1} | gold in top five: {2}" -f $model, $example.Question, ($hits -join ', ')
    }
}
$lines | Tee-Object -FilePath artifacts/phase10/demo-preflight.log
```

The equivalent two-question/all-model API pass was executed for Phase 10 and its output is saved in `artifacts/phase10/demo-preflight.log`. Four model-specific paragraph-vector cache files for this paper were present after warm-up.

## Five-to-seven-minute live walkthrough

1. **0:00-0:40 - Problem.** CiteWise ranks evidence paragraphs within one selected scientific paper; it does not generate scientific answers.
2. **0:40-1:20 - Dataset.** Show QASPER's official paper-level splits and explain that the validation comparison uses 888 scorable questions across 276 papers. Unanswerable and unmatched textual-evidence questions are counted and excluded.
3. **1:20-2:10 - Cleaning and leakage.** Explain stable original paragraph IDs, whitespace/text normalization, train-only logistic fitting and validation-only model selection. The earlier frozen test result is separate and unchanged.
4. **2:10-3:40 - Real retrieval.** Select the example paper/question, run E5, inspect the gold IDs/text, and switch through MiniLM, BGE, GTE, BM25 and logistic. Explain query/passage prefixes and pooling from the model table.
5. **3:40-5:10 - Results.** Open the six-method dashboard. Explain Recall@1/5, MRR@5 and nDCG@5. E5 leads validation point estimates; its paired lead over BGE is uncertain for Recall@1, MRR and nDCG and narrowly supported for Recall@5. MiniLM is the faster, smaller-memory neural choice.
6. **5:10-6:00 - Logistic failure.** The hard-negative training pool has a different BM25 score distribution from full-paper inference candidates; standardization did not resolve the weak validation result.
7. **6:00-7:00 - Accessibility/deployment.** Show the 375/390 CSS-pixel checks, keyboard operation and axe report. State that this is a local Windows/Chromium demo, not public hosting; manual screen-reader and other-browser/physical-device checks remain pending.

## Viva answers

- **Why QASPER?** It provides research-paper questions and mapped textual evidence, allowing paragraph retrieval to be measured against human annotations.
- **Why paper-level splits?** Questions about a paper should not leak across train, validation and test; the official partitions are retained.
- **BM25 versus embeddings?** BM25 scores lexical matches; dense encoders compare query and passage vectors and can match related wording, but may miss exact terms or be affected by truncation.
- **How do the neural models differ?** MiniLM is a compact 6-layer mean-pooled encoder; BGE applies an instruction to short queries and uses CLS pooling; E5 uses asymmetric `query:`/`passage:` inputs and mean pooling; GTE is a general-purpose mean-pooled text embedding model. Their learned spaces and input conventions yield different rankings, but the comparison does not isolate causality.
- **Why choose E5?** It leads all four validation point estimates, has slightly lower measured warm P50 and RSS than BGE, and is configured as the quality-first local default. Its paired advantage over BGE is limited: only Recall@5's unadjusted 95% interval excludes zero.
- **Why is MiniLM faster?** It is the smaller encoder in this set and has shorter measured passage-encoding time and lower peak RSS on the CPU setup. This is a measured system trade-off, not a universal model guarantee.
- **Why did logistic underperform?** Training used a positive-plus-hard-negative candidate sample, while inference ranks all paragraphs. The sampled/full-candidate BM25 distributions differ; standardization did not fix the bounded validation result. This is evidence for a mismatch, not proof of one cause.
- **What does the UI return?** Real ranked source passages, section labels, stable evidence IDs, scores and retrieval timing. It does not generate answers or claim scientific truth.
- **What is not yet verified?** Public HTTPS hosting, Docker build, cross-OS/browser testing and manual assistive-technology testing. Local HTTP, Chromium viewport, keyboard and automated axe evidence are verified.

## Phase 10 verification record

- Local Uvicorn on `127.0.0.1:8765`: `/api/health` returned `ready`; `/api/config` reported all six methods available, E5 as the configured default, and no fallback.
- Both example questions were sent through `/api/search` for all six methods. Each call returned five unique passages, passage text matched the prepared source paper, and every ranking included annotated evidence. The verified ranks are in the table above; repeated warm-up output is in `artifacts/phase10/demo-preflight.log`.
- Python: 44 passed, with one existing Starlette/httpx deprecation warning. Playwright: 11 passed, including the curated demo. Standalone axe: 1 passed with 0 violations. The wheel contains `citewise/index.html` and the six-method validation summary.
- GitHub-ready E5 demo screenshots are copied to `submission/assets/phase10/`; raw browser/test screenshots remain in `artifacts/phase10/ui/` and axe output in `artifacts/phase10/a11y/axe-results.json`.
- The Phase 9 report remains ten pages and unchanged; its ten-browser-test statement is Phase 9 evidence. Phase 10 adds one curated-demo browser case, bringing the current suite to eleven. Phase 10 verifies localhost only; no cloud or Docker deployment is claimed.
