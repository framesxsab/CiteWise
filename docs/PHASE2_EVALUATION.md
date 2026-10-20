# Phase 2: evaluation and model reliability

## Data and decision boundaries

- The source remains `allenai/qasper` at the recorded revision and uses the official paper-level `train`, `validation`, and `test` files.
- Fit the supervised ranker on `train` only. Use `validation` for feature and model decisions. Do not read or optimize against `test` until the configuration is frozen for the final evaluation.
- The earlier `artifacts/evaluation_validation.json` and `artifacts/bge40/evaluation_validation.json` are retained as baseline evidence. Phase 2 writes to new directories and does not replace them.
- QASPER labels identify annotated textual evidence. Results remain retrieval evidence candidates, not validated generated answers.

## Investigate the weak logistic ranker

Run:

```powershell
python -m citewise.ranker_diagnostics --limit 40 --candidate-k 20 --out-dir artifacts/phase2/ranker-top20
```

The command loads the existing artifact for the original raw-feature score, rebuilds sampled pairs from `train`, fits a standardized logistic variant on `train`, and compares four rankings on the first 40 scorable validation questions: BM25, the existing logistic artifact over every paragraph, the standardized refit over every paragraph, and the standardized refit reranking BM25's top 20 candidates. The top-20 cutoff is an experiment setting, not a selected final value.

`ranker_diagnostics.json` records pair balance, feature distributions by class, the difference between sampled training pairs and full-paper candidates, raw and standardized coefficients, and validation metrics. `ranker_experiment_validation.csv` contains per-query outcomes and BM25 shortlist recall; `ranker_failures_validation.csv` contains zero-recall cases; and `ranker_paired_comparisons_validation.csv` contains paper-cluster bootstrap intervals. This diagnostic command does not update the trained artifact. A favorable result on this 40-question sample is not sufficient to claim an improvement.

The current sampler keeps every positive paragraph and at most six highest-BM25 negatives for each scorable training question. Inference scores every paragraph in the paper. The diagnostics quantify the resulting candidate-distribution shift; they do not assume it is the only cause of weak ranking.

## Phase-separated benchmark

Run a bounded, local-cache-only comparison:

```powershell
python -m citewise.benchmark --models bm25 minilm bge-base --limit 40 --out-dir artifacts/phase2/benchmark
python -m citewise.charts --input artifacts/phase2/benchmark/benchmark_validation.json --out-dir artifacts/phase2/benchmark/figures
```

The harness fixes the split to validation and sets `HF_HUB_OFFLINE=1` before model access. A missing cache causes a clear failure; it will not count a network download or retry as inference time. Shared Python runtime imports are timed separately from per-model checkpoint loading. For each selected model it records model load, per-paper paragraph embedding, and query encoding plus ranking after those phases are cached. The JSON records the machine, Python and installed package versions, full and evaluated paper/question counts, per-query metrics and latency, memory readings where the OS exposes them, and measurement scope. On Windows the process peak working set is cumulative across models in the same run.

The harness writes:

- `benchmark_validation.json`: environment, per-phase timings, model summaries, and paired bootstrap results.
- `per_query_validation.csv`: same-query metrics and warm query latency for each model.
- `paired_comparisons_validation.csv`: paired model deltas and 95% confidence intervals.
- `failure_cases_validation.csv`: questions where Recall@5 is zero.
- `figures/`: Recall@5, nDCG@5, warm P95 query latency, and measured setup/embedding time plots.

Bootstrap intervals resample paper IDs with replacement and carry all sampled questions for each selected paper together. The point estimates remain macro means over evaluated questions. All compared models use the same ordered scorable-question sample. The default sample size is 40; a full validation run is explicit (`--limit 888`) and should only be used after the bounded run succeeds and resource use is acceptable.

## Heavy optional models

BGE-M3 and Qwen3-Embedding-0.6B are opt-in. The UI and evaluation harness will use a local Hugging Face cache for these models and will not download them automatically. The benchmark itself is offline for all models. Do not start a large run on the 15.82 GB, CPU-only laptop just because the UI exposes an option.

### Hosted GPU workflow

Use an institution-approved GPU notebook or a user-managed temporary GPU VM with enough RAM and persistent disk. This is an evaluation workflow, not a public deployment. Keep the existing QASPER split files intact and transfer only the project, prepared data, and artifacts needed for the run.

1. Create a clean environment from the project and install `.[models,charts,dev]`. Use a CUDA-enabled PyTorch build compatible with that host's installed driver; record the exact image, driver, Python, PyTorch, and package versions.
2. Confirm `torch.cuda.is_available()` and record the GPU model, RAM, and free disk. Keep the machine private; do not expose an application port for a benchmark job.
3. If a selected checkpoint is not already cached, download only that explicitly selected checkpoint outside the timed benchmark phase. For example:

   ```powershell
   python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B')"
   ```

   These commands intentionally download weights. Do not run them on the local laptop as part of the Phase 2 check.
4. Run the same bounded validation comparison first, then expand to the full validation set only if the GPU, RAM, and disk readings support it:

   ```powershell
   python -m citewise.benchmark --models bm25 minilm bge-base bge-m3 qwen3-0.6b --limit 40 --out-dir artifacts/phase2/gpu40
   python -m citewise.charts --input artifacts/phase2/gpu40/benchmark_validation.json --out-dir artifacts/phase2/gpu40/figures
   ```

5. Copy the JSON/CSV and generated figures back with the hardware and software record. Do not merge results from different query sets into one comparison. Report any failed or unavailable model as not run.

## Freeze gate for final test

The Phase 5 freeze was recorded before test access. The four-method held-out evaluation was executed once during Phase 6 and is stored under `artifacts/phase5/final-test/`; final results are in `submission/CiteWise_Final_Report.pdf`. Do not execute the test command again or tune models, features, cutoffs, thresholds, or selection using those results. The former Phase 2 example command has been retired to avoid a duplicate held-out run.
