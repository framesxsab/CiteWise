"""Build report material directly from completed validation evidence."""
import argparse
import json
from pathlib import Path
from .dataset import QASPER_REVISION, load_papers


def render(root):
    root = Path(root)
    result = json.loads((root / "benchmark_validation.json").read_text(encoding="utf-8"))
    failure = json.loads((root / "failure_analysis.json").read_text(encoding="utf-8"))
    audit_path = root.parent / "accessibility_source.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    browser_path = root.parent / "ui" / "browser_verification_status.json"
    browser = json.loads(browser_path.read_text(encoding="utf-8")) if browser_path.exists() else {}
    neural_models = [r["model"] for r in result["summaries"] if r["model"] not in ("bm25", "supervised-logistic")]
    availability_path = root.parent / "model_availability.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8")) if availability_path.exists() else {}
    target_models = ("minilm", "bge-base", "bge-m3", "qwen3-0.6b")
    target_missing = [name for name in target_models if name not in neural_models]
    fallback_measured = [name for name in neural_models if name in ("e5-base-v2", "gte-base")]
    validation_papers = list(load_papers("data/processed/validation.jsonl"))
    validation_paragraphs = sum(len(p["paragraphs"]) for p in validation_papers)
    validation_question_status = {}
    unsupported_items = 0
    for paper in validation_papers:
        for question in paper["questions"]:
            validation_question_status[question["status"]] = validation_question_status.get(question["status"], 0) + 1
            unsupported_items += question.get("unsupported_figure_items", 0)
    ranker_path = root.parent.parent / "phase2" / "ranker-top20" / "ranker_diagnostics.json"
    ranker = json.loads(ranker_path.read_text(encoding="utf-8")) if ranker_path.exists() else None
    lines = ["# CiteWise: Phase 3 methodology and measured validation results", "",
             "This is methodology/results material for the 10-page faculty report. It is generated from result files, not a claim that the final report, four-neural-model requirement, public deployment or final test is complete.", "",
             "## Research question and contribution", "",
             "How do lexical retrieval, a QASPER-trained lexical classifier and pretrained sentence encoders differ in within-paper evidence recall, rank quality, document preparation cost and warm query latency? CiteWise contributes a reproducible engineering comparison and a documented negative-sampling failure, rather than a new neural architecture or generated-answer truth verifier.", "",
             "## Data and cleaning", "",
             f"Validation uses QASPER revision `{QASPER_REVISION}`: {result['paper_count']} official papers, {validation_paragraphs} non-empty paragraphs and {result['question_count']} questions. The comparison evaluates {result['summaries'][0]['questions_evaluated']} scorable questions in {result['evaluated_paper_count']} papers. Question statuses: `{json.dumps(validation_question_status, sort_keys=True)}`; unsupported figure/table evidence items: {unsupported_items}.", "",
             "Original paper boundaries are retained. Cleaning strips empty paragraphs and collapses whitespace; evidence strings are whitespace-normalized and case-folded before mapping to stable original paragraph IDs. Repeated identical text can map to multiple IDs. Figure/table-only, unmapped and unanswerable annotations are reported rather than invented as textual gold. At least one mapped gold paragraph makes a question scorable. Multiple annotators' mapped textual evidence is unioned. The prepared train/validation files are unchanged; the final test JSONL is not opened.", "",
             "## Models and evaluation protocol", "",
             "The supervised logistic artifact was fitted on train only and is unchanged. Neural weights are pretrained, not fitted on these validation labels. The full original paper paragraphs form each candidate set. Each model sees identical question/gold/candidate IDs and k=5; model-specific input instructions and native tokenizer truncation limits are explicit in the run configuration.", "",
             "Recall@5 divides unique retrieved gold paragraphs by all mapped gold paragraphs. MRR@5 uses the reciprocal rank of the first relevant paragraph, with zero for no hit. nDCG@5 uses binary relevance, logarithmic rank discount and an ideal ranking normalization. Means are macro averages over questions. The 95% intervals use 2,000 paper-cluster bootstrap draws (seed 42), retaining questions together and paired across models. They are validation uncertainty estimates without multiple-comparison correction.", "",
             "Phase 3 applies BGE-base's documented short-query instruction. Phase 2 omitted it, so comparing those runs is a protocol change as well as a sample-size change. Phase 2 remains preserved for its original interpretation.", "",
             "## Measured retrieval quality", "",
             "| Model | Queries | Recall@5 [95% CI] | MRR@5 [95% CI] | nDCG@5 [95% CI] |",
             "|---|---:|---:|---:|---:|"]
    for row in result["summaries"]:
        interval = result["confidence_intervals"][row["model"]]["metrics"]
        cells = [f"{interval[m]['mean']:.4f} [{interval[m]['ci95_low']:.4f}, {interval[m]['ci95_high']:.4f}]"
                 for m in ("recall_at_5", "mrr_at_5", "ndcg_at_5")]
        lines.append(f"| {row['model']} | {row['questions_evaluated']} | " + " | ".join(cells) + " |")
    lines += ["", "## Runtime and memory", "",
              "Each model uses an isolated worker. Runtime imports, local model loading, per-paper document encoding and cached-document query retrieval are timed separately. No network request or download occurs during the benchmark. Wall time includes initialization, audits and checkpoint IO. An interrupted attempt can only retain timing through its last checkpoint; that limitation and the attempt history are in model.json. OS process peak working set is a per-worker high-water mark, not a GPU-memory measure.", "",
              "| Model | Recorded wall s | Imports s | Load s | Document encoding s | Warm p50/p95/p99 ms | Peak RSS GiB |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["summaries"]:
        phase = result["phases"][row["model"]]
        peak = row["peak_memory"].get("peak_rss_bytes")
        peak_text = f"{peak/1024**3:.3f}" if peak else "unmeasured"
        lines.append(f"| {row['model']} | {row['total_execution_seconds']:.2f} | {phase['runtime_initialization_ms']/1000:.2f} | {phase['model_loading_ms']/1000:.2f} | {phase['paragraph_embedding_ms_total']/1000:.2f} | {row['p50_latency_ms']:.2f}/{row['p95_latency_ms']:.2f}/{row['p99_latency_ms']:.2f} | {peak_text} |")
    lines += ["", "These are observed traversal latencies on the recorded host, not a concurrency/load test or a deployment service-level guarantee. Threading differs from Phase 2; do not attribute timing changes solely to model changes.", "",
              "## Paired differences", "", "Differences are model A minus model B in the following table; an interval containing zero does not establish a difference on this validation protocol.", "",
              "| A | B | Metric | Mean difference | 95% paper-cluster CI |", "|---|---|---|---:|---:|"]
    for pair in result["paired_comparisons"]:
        lines.append(f"| {pair['model_a']} | {pair['model_b']} | {pair['metric']} | {pair['mean_difference_a_minus_b']:.4f} | [{pair['ci95_low']:.4f}, {pair['ci95_high']:.4f}] |")
    lines += ["", "## Failure analysis and methodological limitation", "",
              "The unchanged logistic classifier is an original negative result. The Phase 2 train-only diagnostic recorded these candidate-distribution and bounded-validation measurements:", ""]
    if ranker:
        sample = ranker["pair_sampling"]["sampled_pair_features_by_label"]
        valdist = ranker["candidate_diagnostics"]["validation"]["full_candidate_features_by_label"]
        coef = ranker["coefficients"]["existing_raw_artifact"]["coef"]
        lines += [f"Sampled train-pair positive fraction: {ranker['pair_sampling']['sampled_positive_fraction']:.3f} ({ranker['pair_sampling']['positive_pairs']} positive / {ranker['pair_sampling']['negative_pairs']} negative pairs). Mean BM25 on sampled positives/selected hard negatives: {sample['1']['bm25']['mean']:.3f}/{sample['0']['bm25']['mean']:.3f}. Mean BM25 on full validation positives/non-relevant candidates: {valdist['1']['bm25']['mean']:.3f}/{valdist['0']['bm25']['mean']:.3f}. Existing raw BM25/query-coverage coefficients: {coef[0]:.3f}/{coef[1]:.3f}.",
                  f"On the bounded validation comparison, original logistic Recall/MRR/nDCG: {ranker['validation_summaries']['supervised-logistic']['recall_at_5']:.4f}/{ranker['validation_summaries']['supervised-logistic']['mrr_at_5']:.4f}/{ranker['validation_summaries']['supervised-logistic']['ndcg_at_5']:.4f}; standardized refit: {ranker['validation_summaries']['standardized-logistic']['recall_at_5']:.4f}/{ranker['validation_summaries']['standardized-logistic']['mrr_at_5']:.4f}/{ranker['validation_summaries']['standardized-logistic']['ndcg_at_5']:.4f}. The BM25 top-20 rerank result and CI are in ranker_diagnostics.json.", ""]
    lines += ["The observed sign reversal is consistent with training on a candidate pool of positives plus only the hardest BM25 negatives, then ranking every paragraph at inference. It is a measured distribution mismatch, not proof that one feature alone caused every failure. Scaling alone did not change the bounded result; no experimental ranker replaces the saved baseline.", "",
              "Full-validation lexical-versus-neural disagreement counts:", "", "```json", json.dumps(failure["comparisons"], indent=2), "```", "",
              "The failure export includes exact questions, gold and returned passage text, lexical coverage and token-cap flags. Terminology mismatch and semantic ambiguity require qualitative reading of those examples; observable token overlap or truncation is not itself a causal diagnosis. Questions without mapped evidence remain excluded and are exported separately.", "",
              "## Reproducibility and submission boundary", "",
              f"Protocol: `{result['protocol']}`. Full configurations, immutable snapshot commit IDs, dataset/query-plan hashes, checkpoint identifiers, package versions and hardware are in benchmark_validation.json and checksummed model checkpoints. Figures are generated from these files only. See docs/PHASE3_PROTOCOL.md for exact resume, figure, failure-analysis and hosted/fallback commands.", "",
              f"Recorded host: {result['configurations'][neural_models[0]]['hardware_and_software']['os']}; Python {result['configurations'][neural_models[0]]['hardware_and_software']['python']}; {result['configurations'][neural_models[0]]['hardware_and_software']['cpu_count']} CPUs; {result['configurations'][neural_models[0]]['hardware_and_software']['total_physical_memory_bytes']/1024**3:.2f} GiB physical RAM; CUDA {result['configurations'][neural_models[0]]['hardware_and_software']['cuda_available']}. Exact package versions are in the configuration JSON.", "",
              f"Measured neural models: {', '.join(neural_models)}. The original target set is MiniLM, BGE-base, BGE-M3 and Qwen3-Embedding-0.6B; still unmeasured: {', '.join(target_missing) if target_missing else 'none'}. Fallback models actually measured: {', '.join(fallback_measured) if fallback_measured else 'none'}. E5/GTE remain an explicitly labeled older-model fallback, not a substitute for claiming the target comparison complete. Availability details: `{availability.get('download_preflight', {}).get('error', 'not recorded')}`. No borrowed scores fill missing experiments.", "",
              f"Automated source accessibility audit passed: {audit.get('passed', 'unavailable')}. Browser verification status: {browser.get('status', 'not recorded')}. The live API/content comparison and specific browser limitations are recorded separately under artifacts/phase3/ui. Earlier screenshots remain Phase 2 evidence. Public deployment and cross-platform execution are not established.", "",
              "Model selection remains provisional until the additional-model comparison and validation decisions are complete. Freeze preprocessing, model configuration and selection before the final held-out test. Do not use final test outcomes for tuning.", "",
              "## Report assembly", "", "Use REPORT_BLUEPRINT.md for the 10-page allocation and LITERATURE_COMPARISON.md for six primary-source-backed recent papers. Insert the measured quality, interval, timing and latency-distribution plots from this output directory. Use the failure review queue for transparent case studies; do not label assistant interpretation as independent human evaluation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/phase3/full-validation-v1")
    parser.add_argument("--output", default="docs/REPORT_METHODS_RESULTS.md")
    args = parser.parse_args()
    Path(args.output).write_text(render(args.out_dir), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
