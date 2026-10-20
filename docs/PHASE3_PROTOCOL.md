# Phase 3 research protocol

This phase uses the existing real QASPER validation JSONL only. No final test paper or question is loaded. Train remains the only fitting split, and the existing supervised logistic artifact is evaluated without retraining or replacement.

## Frozen comparison

- Protocol identifier: `qasper-paragraph-retrieval-v3-instructed`.
- Dataset: official validation split, 281 papers, 1,005 questions; 888 mapped scorable questions in 276 papers. All models receive the same ordered question plan, gold paragraph IDs, full original paper candidate pool and `k=5`.
- Exclusions: 60 unanswerable questions and 57 questions with unmatched/no mapped textual evidence. Figure/table-only annotations do not become invented textual evidence. Existing normalization, duplicate text mapping and original splits are preserved.
- Quality: macro means over scorable questions for Recall@5, MRR@5 and binary nDCG@5. Native tokenizer length limits are explicit; original paragraphs are not split, padded with synthetic text or replaced.
- Uncertainty: 2,000 percentile bootstrap replicates, seed 42, resampling papers with all their questions together. Both per-model intervals and paired differences are exported. These are validation intervals, without multiple-comparison correction, and do not constitute final test evidence.
- CPU measurement: one model per fresh worker process, float32, four Torch threads, one OpenBLAS thread, batch size 16. GPU runs must report their separate device/configuration; hardware-dependent latency must not be pooled across hosts.
- Model runtime imports, local checkpoint loading, paragraph encoding, tokenizer-length auditing and warm query encoding/scoring/sorting are recorded separately. Model checkpoints are cache-only; no network transfer or retry belongs to timed inference.
- Peak memory is the OS process working-set high-water mark of that model's worker, rather than a shared multi-model process. Other applications still affect the available RAM and runtime; observed latencies are a single measured traversal, not a service-load benchmark.

## Model-specific inputs and provenance

| Model | Token limit | Pooling | Query input | Document input | Phase 3 availability |
|---|---:|---|---|---|---|
| all-MiniLM-L6-v2 | 256 | attention-masked mean | plain text | plain text | cached locally |
| BGE-base-en-v1.5 | 512 | CLS | documented retrieval instruction plus question | plain text | cached locally |
| BGE-M3 | 8,192 | CLS dense vector | plain text | plain text | not cached; GPU workflow pending |
| Qwen3-Embedding-0.6B | 32,768 | last token | explicit scientific-evidence task instruction | plain text | not cached; GPU workflow pending |
| E5-base-v2 | 512 | attention-masked mean | `query: ` prefix | `passage: ` prefix | optional older CPU alternative; not cached |
| GTE-base | 512 | attention-masked mean | plain text | plain text | optional older CPU alternative; not cached |

All vectors are L2-normalized and ranked by normalized dot product. Library default prompts are suppressed to avoid double-prefixing. Pooling is checked against the loaded runtime configuration, with support for legacy and current SentenceTransformers config formats. The model snapshot directory's immutable commit ID is recorded. E5/GTE and the two larger target models have input handling implemented, but no measured scores are claimed without successful execution.

The Phase 2 BGE-base run omitted the recommended short-query retrieval instruction. Phase 3 applies it, so its full-validation result is a new protocol, not evidence that merely adding more questions caused a quality change. The earlier Phase 2 artifacts remain available for their original comparison.

Primary model cards, checked 2026-10-18: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE-base](https://huggingface.co/BAAI/bge-base-en-v1.5), [BGE-M3](https://huggingface.co/BAAI/bge-m3), [Qwen3](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [E5](https://huggingface.co/intfloat/e5-base-v2), [GTE](https://huggingface.co/thenlper/gte-base). E5/GTE are older comparison models; do not label them the latest state of the art. For Qwen the task text is: `Given a question about a scientific paper, retrieve passages that provide evidence for answering it.`

## Run and resume

Use the virtual-environment interpreter directly if PowerShell activation is disabled:

```powershell
.\.venv\Scripts\python.exe -u -m citewise.research_benchmark --models bm25 supervised-logistic minilm bge-base --limit 888 --threads 4 --batch-size 16 --out-dir artifacts/phase3/full-validation-v1 *> artifacts/phase3/full-validation-v1/console.log
```

Each completed paper is written atomically with a checksum. Dataset bytes, query plan, retrieval/metric/worker source, model snapshot, environment and configuration identify the checkpoint. A changed configuration is refused; use a new output directory instead of editing checkpoints. OS file locks prevent two workers writing the same model concurrently. An interrupted attempt loses at most its unfinished paper; no completed query is filled with a synthetic score. The Phase 3 run is complete and preserved. Phase 4 changed retrieval source, so do not resume `full-validation-v1` using the current tree; the checkpoint source hash will reject it. Use a new output directory for subsequent validation experiments.

Attempt timings include runtime setup, checkpoint writes and progress logging. After a hard kill, work since the last saved checkpoint cannot be measured retrospectively; that limit is recorded. The Phase 3 run exercised actual interruption and resume during MiniLM. Progress output is redirected to a file to avoid console backpressure.

```powershell
.\.venv\Scripts\python.exe -m citewise.charts --input artifacts/phase3/full-validation-v1/benchmark_validation.json --out-dir artifacts/phase3/full-validation-v1/figures
.\.venv\Scripts\python.exe -m citewise.failure_analysis --out-dir artifacts/phase3/full-validation-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m citewise.accessibility_audit
```

## Additional-model execution blocker and GPU path

The current execution environment denies outbound Hugging Face requests (`WinError 10013`). Only MiniLM and BGE-base are in the local cache. No hosted GPU connection has been supplied. This blocks both the larger target-model experiment and the smaller fallback experiment; no score is substituted from a published leaderboard.

BGE-M3 has roughly 569 million parameters; Qwen3 has roughly 600 million. Float32 weights alone require approximately 2.12/2.24 GiB respectively, before activations, tokenizer/runtime and cache memory. Those estimates are lower bounds, not measured peak memory. This laptop also encountered limited Windows commit capacity. Neither larger model is initialized locally in this phase.

On an already authorized, network-accessible GPU host, follow the installation/resource preflight in `PHASE2_EVALUATION.md`, stage only the explicitly selected models outside timing, and begin with batch size 1 and a small validation limit. Record actual GPU/RAM/peak-memory measurements before expanding. For the full target comparison:

```sh
python -m citewise.research_benchmark --models bm25 minilm bge-base bge-m3 qwen3-0.6b --limit 888 --device cuda --batch-size 1 --threads 4 --out-dir artifacts/phase3/gpu-validation
```

If no GPU host is available, stage the smaller fallback checkpoints from a network-accessible terminal. Avoid downloading duplicate framework formats:

```powershell
python -c "from huggingface_hub import snapshot_download; files=['*.json','model.safetensors','vocab.txt']; snapshot_download('intfloat/e5-base-v2',allow_patterns=files); snapshot_download('thenlper/gte-base',allow_patterns=files)"
```

Then add them to the same validation comparison with `--models bm25 supervised-logistic minilm bge-base e5-base-v2 gte-base`, using the same configuration/output directory and unedited benchmark source. Completed existing models resume without inference. The fallback is a documented deviation from the original four-neural-model plan, not equivalent to measuring BGE-M3/Qwen3.

## UI and submission evidence boundary

Phase 3 adds explicit empty/loading states, locks form controls during retrieval, resets stale annotated questions, labels optional/fallback models and fixes the source palette's muted-text contrast. The automated source audit checks labels, landmarks, document metadata, live status, focus styling and declared color pairs. It cannot establish computed-style accessibility, actual focus behavior or mobile layout. Browser control currently times out on page operations; fresh axe results/screenshots must remain blocked until that connection works. Existing Phase 2 screenshots are earlier local evidence.

This Phase 3 record predates the held-out evaluation. The later Phase 5 freeze and single Phase 6 test run are documented in `docs/PHASE5_FREEZE.md` and `submission/CiteWise_Final_Report.pdf`. Public deployment, cross-platform execution and assistive-technology validation are not inferred from local HTTP responses.
