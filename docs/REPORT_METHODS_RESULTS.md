# CiteWise: Phase 3 methodology and measured validation results

This is a historical Phase 3 validation-only methods/results snapshot generated before held-out testing. The final test results are reported in `submission/CiteWise_Final_Report.pdf`; the four-neural-model and public-deployment limitations remain explicit.

## Research question and contribution

How do lexical retrieval, a QASPER-trained lexical classifier and pretrained sentence encoders differ in within-paper evidence recall, rank quality, document preparation cost and warm query latency? CiteWise contributes a reproducible engineering comparison and a documented negative-sampling failure, rather than a new neural architecture or generated-answer truth verifier.

## Data and cleaning

Validation uses QASPER revision `91bc63e221475d59a6e17e4299fbfc5a919d7c7a`: 281 official papers, 13266 non-empty paragraphs and 1005 questions. The comparison evaluates 888 scorable questions in 276 papers. Question statuses: `{"scorable": 888, "unanswerable": 60, "unmatched_or_no_text_evidence": 57}`; unsupported figure/table evidence items: 253.

Original paper boundaries are retained. Cleaning strips empty paragraphs and collapses whitespace; evidence strings are whitespace-normalized and case-folded before mapping to stable original paragraph IDs. Repeated identical text can map to multiple IDs. Figure/table-only, unmapped and unanswerable annotations are reported rather than invented as textual gold. At least one mapped gold paragraph makes a question scorable. Multiple annotators' mapped textual evidence is unioned. The prepared train/validation files are unchanged. This Phase 3 snapshot did not open the final test JSONL; the later held-out evaluation is separate.

## Models and evaluation protocol

The supervised logistic artifact was fitted on train only and is unchanged. Neural weights are pretrained, not fitted on these validation labels. The full original paper paragraphs form each candidate set. Each model sees identical question/gold/candidate IDs and k=5; model-specific input instructions and native tokenizer truncation limits are explicit in the run configuration.

Recall@5 divides unique retrieved gold paragraphs by all mapped gold paragraphs. MRR@5 uses the reciprocal rank of the first relevant paragraph, with zero for no hit. nDCG@5 uses binary relevance, logarithmic rank discount and an ideal ranking normalization. Means are macro averages over questions. The 95% intervals use 2,000 paper-cluster bootstrap draws (seed 42), retaining questions together and paired across models. They are validation uncertainty estimates without multiple-comparison correction.

Phase 3 applies BGE-base's documented short-query instruction. Phase 2 omitted it, so comparing those runs is a protocol change as well as a sample-size change. Phase 2 remains preserved for its original interpretation.

## Measured retrieval quality

| Model | Queries | Recall@5 [95% CI] | MRR@5 [95% CI] | nDCG@5 [95% CI] |
|---|---:|---:|---:|---:|
| bm25 | 888 | 0.4432 [0.4104, 0.4744] | 0.3456 [0.3201, 0.3721] | 0.3286 [0.3033, 0.3532] |
| supervised-logistic | 888 | 0.0493 [0.0374, 0.0627] | 0.0520 [0.0413, 0.0638] | 0.0391 [0.0309, 0.0486] |
| minilm | 888 | 0.4611 [0.4314, 0.4899] | 0.3605 [0.3334, 0.3891] | 0.3457 [0.3220, 0.3695] |
| bge-base | 888 | 0.5210 [0.4920, 0.5508] | 0.4387 [0.4099, 0.4660] | 0.4142 [0.3898, 0.4395] |

## Runtime and memory

Each model uses an isolated worker. Runtime imports, local model loading, per-paper document encoding and cached-document query retrieval are timed separately. No network request or download occurs during the benchmark. Wall time includes initialization, audits and checkpoint IO. An interrupted attempt can only retain timing through its last checkpoint; that limitation and the attempt history are in model.json. OS process peak working set is a per-worker high-water mark, not a GPU-memory measure.

| Model | Recorded wall s | Imports s | Load s | Document encoding s | Warm p50/p95/p99 ms | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 5.86 | 0.17 | 0.00 | 0.00 | 2.96/6.17/15.75 | 0.057 |
| supervised-logistic | 13.82 | 0.18 | 4.29 | 0.00 | 6.10/11.30/25.96 | 0.182 |
| minilm | 607.90 | 62.53 | 1.03 | 518.98 | 20.07/26.33/32.00 | 0.739 |
| bge-base | 4155.97 | 27.56 | 1.36 | 4036.48 | 92.48/127.67/141.83 | 1.294 |

These are observed traversal latencies on the recorded host, not a concurrency/load test or a deployment service-level guarantee. Threading differs from Phase 2; do not attribute timing changes solely to model changes.

## Paired differences

Differences are model A minus model B in the following table; an interval containing zero does not establish a difference on this validation protocol.

| A | B | Metric | Mean difference | 95% paper-cluster CI |
|---|---|---|---:|---:|
| bm25 | supervised-logistic | recall_at_5 | 0.3939 | [0.3609, 0.4297] |
| bm25 | supervised-logistic | mrr_at_5 | 0.2936 | [0.2650, 0.3224] |
| bm25 | supervised-logistic | ndcg_at_5 | 0.2895 | [0.2638, 0.3169] |
| bm25 | minilm | recall_at_5 | -0.0179 | [-0.0516, 0.0173] |
| bm25 | minilm | mrr_at_5 | -0.0149 | [-0.0431, 0.0147] |
| bm25 | minilm | ndcg_at_5 | -0.0171 | [-0.0428, 0.0103] |
| bm25 | bge-base | recall_at_5 | -0.0778 | [-0.1104, -0.0450] |
| bm25 | bge-base | mrr_at_5 | -0.0931 | [-0.1189, -0.0668] |
| bm25 | bge-base | ndcg_at_5 | -0.0856 | [-0.1094, -0.0607] |
| supervised-logistic | minilm | recall_at_5 | -0.4118 | [-0.4430, -0.3813] |
| supervised-logistic | minilm | mrr_at_5 | -0.3085 | [-0.3356, -0.2796] |
| supervised-logistic | minilm | ndcg_at_5 | -0.3066 | [-0.3308, -0.2814] |
| supervised-logistic | bge-base | recall_at_5 | -0.4717 | [-0.5031, -0.4409] |
| supervised-logistic | bge-base | mrr_at_5 | -0.3867 | [-0.4164, -0.3562] |
| supervised-logistic | bge-base | ndcg_at_5 | -0.3751 | [-0.4004, -0.3489] |
| minilm | bge-base | recall_at_5 | -0.0599 | [-0.0861, -0.0337] |
| minilm | bge-base | mrr_at_5 | -0.0782 | [-0.1024, -0.0568] |
| minilm | bge-base | ndcg_at_5 | -0.0685 | [-0.0889, -0.0500] |

## Failure analysis and methodological limitation

The unchanged logistic classifier is an original negative result. The Phase 2 train-only diagnostic recorded these candidate-distribution and bounded-validation measurements:

Sampled train-pair positive fraction: 0.223 (3603 positive / 12581 negative pairs). Mean BM25 on sampled positives/selected hard negatives: 3.329/5.044. Mean BM25 on full validation positives/non-relevant candidates: 3.055/1.704. Existing raw BM25/query-coverage coefficients: -0.126/-2.013.
On the bounded validation comparison, original logistic Recall/MRR/nDCG: 0.0175/0.0333/0.0182; standardized refit: 0.0175/0.0333/0.0182. The BM25 top-20 rerank result and CI are in ranker_diagnostics.json.

The observed sign reversal is consistent with training on a candidate pool of positives plus only the hardest BM25 negatives, then ranking every paragraph at inference. It is a measured distribution mismatch, not proof that one feature alone caused every failure. Scaling alone did not change the bounded result; no experimental ranker replaces the saved baseline.

Full-validation lexical-versus-neural disagreement counts:

```json
{
  "minilm": {
    "bm25_better": 178,
    "gold_paragraph_exceeds_token_cap": 144,
    "low_lexical_overlap_candidate_review": 140,
    "short_query_candidate_review": 169,
    "tie": 492,
    "neural_better": 218
  },
  "bge-base": {
    "tie": 520,
    "gold_paragraph_exceeds_token_cap": 13,
    "low_lexical_overlap_candidate_review": 140,
    "short_query_candidate_review": 169,
    "neural_better": 240,
    "bm25_better": 128
  }
}
```

The failure export includes exact questions, gold and returned passage text, lexical coverage and token-cap flags. Terminology mismatch and semantic ambiguity require qualitative reading of those examples; observable token overlap or truncation is not itself a causal diagnosis. Questions without mapped evidence remain excluded and are exported separately.

## Reproducibility and submission boundary

Protocol: `qasper-paragraph-retrieval-v3-instructed`. Full configurations, immutable snapshot commit IDs, dataset/query-plan hashes, checkpoint identifiers, package versions and hardware are in benchmark_validation.json and checksummed model checkpoints. Figures are generated from these files only. See docs/PHASE3_PROTOCOL.md for exact resume, figure, failure-analysis and hosted/fallback commands.

Recorded host: Windows-11-10.0.26200-SP0; Python 3.14.3; 16 CPUs; 15.82 GiB physical RAM; CUDA False. Exact package versions are in the configuration JSON.

Measured neural models: minilm, bge-base. The original target set is MiniLM, BGE-base, BGE-M3 and Qwen3-Embedding-0.6B; still unmeasured: bge-m3, qwen3-0.6b. Fallback models actually measured: none. E5/GTE remain an explicitly labeled older-model fallback, not a substitute for claiming the target comparison complete. Availability details: `URLError: WinError 10013, socket access forbidden`. No borrowed scores fill missing experiments.

Automated source accessibility audit passed: True. Browser verification status: BLOCKED. The live API/content comparison and specific browser limitations are recorded separately under artifacts/phase3/ui. Earlier screenshots remain Phase 2 evidence. Public deployment and cross-platform execution are not established.

This Phase 3 snapshot records validation-stage model decisions. Final model selection and the single held-out evaluation are in the Phase 5 freeze and Phase 6 report. Do not use test outcomes for tuning.

## Report assembly

Use REPORT_BLUEPRINT.md for the 10-page allocation and LITERATURE_COMPARISON.md for six primary-source-backed recent papers. Insert the measured quality, interval, timing and latency-distribution plots from this output directory. Use the failure review queue for transparent case studies; do not label assistant interpretation as independent human evaluation.
