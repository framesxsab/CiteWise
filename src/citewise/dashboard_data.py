"""Build the UI's validation-only aggregate from saved QASPER evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .benchmark import paired_bootstrap
from .dataset import load_papers
from .evaluation import metrics_for_ranking
from .research_benchmark import METRICS, cluster_intervals
from .retrieval import MODEL_IDS, MODEL_REVISIONS, MODEL_SPECS

MODEL_INFO = {
    "bm25": ("BM25", "lexical baseline", "Classical lexical paragraph retrieval."),
    "supervised-logistic": ("Supervised logistic", "supervised lexical baseline",
                            "Train-fitted feature ranker; retained as a measured negative result."),
    "minilm": ("MiniLM-L6-v2", "neural retriever", "Pretrained sentence encoder with mean pooling and 256-token truncation."),
    "bge-base": ("BGE-base-en-v1.5", "neural retriever", "Pretrained instructed encoder with CLS pooling and 512-token truncation."),
    "e5-base-v2": ("E5-base-v2", "neural retriever", "Uses query: and passage: prefixes with mean pooling and 512-token truncation."),
    "gte-base": ("GTE-base", "neural retriever", "Uses plain text with mean pooling and 512-token truncation."),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _recall_at_1(rows, reps=2000, seed=42):
    import numpy as np

    by_paper = defaultdict(list)
    for row in rows:
        gold = {value for value in row["gold_ids"].split("|") if value}
        ranked = [value for value in row["retrieved_ids"].split("|") if value]
        if not gold or not ranked:
            raise ValueError("Scorable validation rows need gold IDs and at least one retrieved passage")
        by_paper[row["paper_id"]].append(float(ranked[0] in gold) / len(gold))
    papers = sorted(by_paper)
    counts = np.array([len(by_paper[paper]) for paper in papers], dtype=np.int64)
    sums = np.array([sum(by_paper[paper]) for paper in papers], dtype=np.float64)
    weights = np.random.default_rng(seed).multinomial(
        len(papers), np.full(len(papers), 1 / len(papers)), size=reps
    )
    samples = (weights @ sums) / (weights @ counts)
    return {"mean": float(sums.sum() / counts.sum()),
            "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)),
            "paper_clusters": len(papers), "bootstrap_reps": reps, "seed": seed}


def build_summary(benchmark_path, predictions_path, audit_path, output_path, reps=2000, seed=42,
                  validation_path=None, comparison_csv_path=None):
    benchmark_path, predictions_path, audit_path = map(Path, (benchmark_path, predictions_path, audit_path))
    result = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if result.get("split") != "validation":
        raise ValueError("The application comparison bundle only accepts validation results")
    if result.get("test_split_accessed", False):
        raise ValueError("The application comparison bundle refuses results that accessed test data")
    dataset_revision = result.get("dataset_revision") or audit.get("revision")
    if audit.get("revision") != dataset_revision:
        raise ValueError("Evaluation and data-audit revisions differ")

    by_model = defaultdict(list)
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_model[row["model"]].append(row)
    summaries = {row["model"]: row for row in result.get("summaries", [])}
    if set(by_model) != set(summaries) or not summaries:
        raise ValueError("Saved per-query rows and benchmark summaries have different model sets")
    configurations = result.get("configurations", {})
    if set(configurations) != set(summaries):
        raise ValueError("Saved model configurations and result summaries have different model sets")
    data_hashes = {cfg.get("data_sha256") for cfg in configurations.values() if cfg.get("data_sha256")}
    if len(data_hashes) > 1:
        raise ValueError("Models were evaluated against different dataset files")
    shared_config = ("protocol", "data_sha256", "query_plan_sha256", "normalize_embeddings",
                     "precision", "similarity", "batch_size", "threads", "device", "blas_threads", "limit", "k")
    first_config = configurations[next(iter(summaries))]
    if result.get("protocol") != first_config.get("protocol") or any(
            any(config.get(key) != first_config.get(key) for key in shared_config)
            for config in configurations.values()):
        raise ValueError("Model configurations differ in dataset, query plan, or evaluation protocol")
    for name, config in configurations.items():
        if config.get("model") != name or summaries[name].get("checkpoint_id") != _object_digest(config):
            raise ValueError(f"Configuration fingerprint mismatch for {name}")

    source_rows = None
    if validation_path is not None:
        validation_path = Path(validation_path)
        if _digest(validation_path) not in data_hashes:
            raise ValueError("Validation source data does not match the benchmark data hash")
        source_rows = {}
        paper_map = {paper["id"]: paper for paper in load_papers(validation_path)}
        expected = {}
        for paper in paper_map.values():
            for question in paper["questions"]:
                if question["status"] == "scorable":
                    expected[(paper["id"], question["id"])] = (paper, question)
        query_limit = first_config.get("limit")
        expected_items = list(expected.items())[:query_limit] if query_limit else list(expected.items())
        expected = dict(expected_items)
        if len(expected) != audit["splits"]["validation"]["questions_scorable"]:
            raise ValueError("Validation question coverage differs from the saved audit")
        planned_query_hash = _object_digest([
            (paper_id, question_id, question["gold_paragraph_ids"])
            for (paper_id, question_id), (_, question) in expected.items()
        ])
        if any(config.get("query_plan_sha256") != planned_query_hash for config in configurations.values()):
            raise ValueError("Benchmark query plan does not match the official validation records")
        source_rows = expected
    query_sets = {}
    paired_rows = []
    for name, rows in by_model.items():
        keys = [(row["paper_id"], row["question_id"]) for row in rows]
        if len(keys) != len(set(keys)) or len(rows) != summaries[name]["questions_evaluated"]:
            raise ValueError(f"Duplicate or incomplete validation predictions for {name}")
        query_sets[name] = set(keys)
        for row in rows:
            gold = [item for item in row["gold_ids"].split("|") if item]
            ranked = [item for item in row["retrieved_ids"].split("|") if item]
            if not gold or not ranked or len(ranked) > 5 or len(ranked) != len(set(ranked)):
                raise ValueError(f"Invalid gold evidence or ranking for {name}: {row['question_id']}")
            if source_rows is not None:
                item = source_rows.get((row["paper_id"], row["question_id"]))
                if item is None:
                    raise ValueError(f"Prediction is not an official scorable validation question: {row['question_id']}")
                paper, question = item
                if gold != question["gold_paragraph_ids"]:
                    raise ValueError(f"Gold evidence mismatch for {name}: {row['question_id']}")
                valid_ids = {paragraph["id"] for paragraph in paper["paragraphs"]}
                if not set(ranked) <= valid_ids:
                    raise ValueError(f"Retrieved evidence is not in its source paper for {name}: {row['question_id']}")
            scores = metrics_for_ranking(ranked, gold, 5)
            for key, actual in scores.items():
                if not math.isclose(float(row[f"{key}_at_5"]), actual, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"Saved per-query {key}@5 does not match evidence IDs for {name}")
                row[f"{key}_at_5"] = actual
            row["recall_at_1"] = metrics_for_ranking(ranked[:1], gold, 1)["recall"]
            paired_rows.append(row)
    if len({frozenset(rows) for rows in query_sets.values()}) != 1:
        raise ValueError("Model predictions do not cover identical validation questions")
    if source_rows is not None and next(iter(query_sets.values())) != set(source_rows):
        raise ValueError("Model predictions do not cover the planned scorable validation questions")

    for name, rows in by_model.items():
        summary = summaries[name]
        for key, field in (("recall_at_5", "recall_at_k"), ("mrr_at_5", "mrr_at_k"), ("ndcg_at_5", "ndcg_at_k")):
            mean = statistics.mean(float(row[key]) for row in rows)
            if not math.isclose(mean, summary[field], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Saved aggregate {field} does not match per-query rows for {name}")
        recomputed = cluster_intervals(rows, reps=reps, seed=seed)
        stored = result.get("confidence_intervals", {}).get(name, {})
        if (recomputed["paper_clusters"] != stored.get("paper_clusters")
                or recomputed["query_count"] != stored.get("query_count")):
            raise ValueError(f"Saved confidence-interval scope does not match predictions for {name}")
        for metric in METRICS:
            for bound in ("mean", "ci95_low", "ci95_high"):
                if not math.isclose(recomputed["metrics"][metric][bound],
                                    stored.get("metrics", {}).get(metric, {}).get(bound, float("nan")),
                                    rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Saved {metric} bootstrap {bound} does not match predictions for {name}")

    paired = result.get("paired_comparisons", [])
    if len(paired) != len(summaries) * (len(summaries) - 1) // 2 * len(METRICS):
        raise ValueError("Saved paired-comparison rows are incomplete")
    paired_keys = set()
    for pair in paired:
        if (pair.get("query_count") != len(next(iter(query_sets.values())))
                or pair.get("paper_clusters") != len({paper for paper, _ in next(iter(query_sets.values()))})
                or pair.get("bootstrap_reps") != reps):
            raise ValueError("Saved paired-comparison scope does not match validation predictions")
        key = (pair.get("model_a"), pair.get("model_b"), pair.get("metric"))
        if key in paired_keys or pair.get("metric") not in METRICS:
            raise ValueError("Saved paired-comparison identities are invalid or duplicated")
        paired_keys.add(key)
        recalculated = paired_bootstrap(paired_rows, pair["model_a"], pair["model_b"],
                                        pair["metric"], reps, seed)
        for field in ("mean_difference_a_minus_b", "ci95_low", "ci95_high"):
            if not math.isclose(recalculated[field], pair[field], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Saved paired bootstrap {field} does not match predictions")

    neural_models = [name for name in summaries if MODEL_INFO.get(name, (None, "", ""))[1] == "neural retriever"]
    recall_at_1_pairs = [paired_bootstrap(paired_rows, left, right, "recall_at_1", reps, seed)
                         for index, left in enumerate(neural_models) for right in neural_models[index + 1:]]

    models = []
    for name, summary in summaries.items():
        label, family, description = MODEL_INFO.get(name, (name, "retriever", ""))
        config = configurations.get(name, {})
        intervals = result.get("confidence_intervals", {}).get(name, {}).get("metrics", {})
        scores = {"recall_at_1": _recall_at_1(by_model[name], reps=reps, seed=seed)}
        for source, target in (("recall_at_k", "recall_at_5"),
                               ("mrr_at_k", "mrr_at_5"),
                               ("ndcg_at_k", "ndcg_at_5")):
            ci = intervals.get(target, {})
            scores[target] = {"mean": summary[source], "ci95_low": ci.get("ci95_low"),
                              "ci95_high": ci.get("ci95_high")}
        models.append({
            "id": name, "label": label, "family": family, "description": description,
            "status": "evaluated", "question_count": summary["questions_evaluated"],
            "scores": scores,
            "latency_ms": {"p50": summary["p50_latency_ms"], "p95": summary["p95_latency_ms"],
                           "basis": summary.get("latency_basis", "not recorded")},
            "paper_count": len({paper for paper, _ in query_sets[name]}),
            "model_loading_ms": result.get("phases", {}).get(name, {}).get("model_loading_ms"),
            "document_encoding_ms": result.get("phases", {}).get(name, {}).get("paragraph_embedding_ms_total"),
            "peak_rss_bytes": summary.get("peak_memory", {}).get("peak_rss_bytes"),
            "model_id": config.get("model_id"), "model_revision": config.get("model_revision"),
            "spec": config.get("encoder"), "checkpoint_id": summary.get("checkpoint_id"),
        })

    for name in ("e5-base-v2", "gte-base"):
        if name not in summaries:
            label, family, description = MODEL_INFO[name]
            models.append({"id": name, "label": label, "family": family,
                           "description": description, "status": "not_evaluated",
                           "reason": "No validation predictions or aggregate score exist for this model.",
                           "model_id": MODEL_IDS[name], "model_revision": MODEL_REVISIONS[name],
                           "spec": MODEL_SPECS[name]})

    validation_counts = audit["splits"]["validation"]
    question_count = next(iter(summaries.values()))["questions_evaluated"]
    if question_count != validation_counts["questions_scorable"]:
        raise ValueError("Evaluation coverage does not match the saved validation audit")
    query_keys = next(iter(query_sets.values()))
    bundle = {
        "dataset": {"name": audit["source"], "revision": audit["revision"], "license": audit["license"],
                    "splits": audit["splits"]},
        "evaluation": {"split": "validation", "protocol": result.get("protocol"),
                       "paper_count": result.get("paper_count"),
                       "evaluated_paper_clusters": len({paper for paper, _ in query_keys}),
                       "question_count": question_count, "top_k": 5,
                       "total_questions": validation_counts["questions"],
                       "configuration": {key: first_config.get(key) for key in
                                         ("protocol", "device", "threads", "batch_size", "precision",
                                          "normalize_embeddings", "similarity", "blas_threads")},
                       "environment": first_config.get("hardware_and_software", {}),
                       "source_compatibility": result.get("source_compatibility"),
                       "bootstrap": {"unit": "paper", "replicates": reps, "seed": seed},
                       "models": models,
                       "paired_comparisons": result["paired_comparisons"],
                       "neural_recall_at_1_comparisons": recall_at_1_pairs,
                       "test_split_accessed": bool(result.get("test_split_accessed", False))},
        "source_artifacts": {
            "benchmark": {"path": str(benchmark_path).replace("\\", "/"), "sha256": _digest(benchmark_path)},
            "per_query_predictions": {"path": str(predictions_path).replace("\\", "/"), "sha256": _digest(predictions_path)},
            "data_audit": {"path": str(audit_path).replace("\\", "/"), "sha256": _digest(audit_path)},
        },
    }
    Path(output_path).write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if comparison_csv_path is not None:
        comparison_csv_path = Path(comparison_csv_path)
        comparison_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["model", "family", "model_id", "model_revision", "question_count", "paper_count",
                  "recall_at_1", "recall_at_1_ci95_low", "recall_at_1_ci95_high",
                  "recall_at_5", "recall_at_5_ci95_low", "recall_at_5_ci95_high",
                  "mrr_at_5", "mrr_at_5_ci95_low", "mrr_at_5_ci95_high",
                  "ndcg_at_5", "ndcg_at_5_ci95_low", "ndcg_at_5_ci95_high",
                  "warm_p50_ms", "warm_p95_ms", "model_loading_ms", "document_encoding_ms",
                  "peak_rss_bytes", "checkpoint_id"]
        with comparison_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for model in models:
                row = {key: model.get(key) for key in ("id", "family", "model_id", "model_revision",
                                                       "question_count", "paper_count", "model_loading_ms",
                                                       "document_encoding_ms", "peak_rss_bytes", "checkpoint_id")}
                row["model"] = row.pop("id")
                row["warm_p50_ms"] = model.get("latency_ms", {}).get("p50")
                row["warm_p95_ms"] = model.get("latency_ms", {}).get("p95")
                for metric in ("recall_at_1", "recall_at_5", "mrr_at_5", "ndcg_at_5"):
                    score = model.get("scores", {}).get(metric, {})
                    row[metric] = score.get("mean")
                    row[f"{metric}_ci95_low"] = score.get("ci95_low")
                    row[f"{metric}_ci95_high"] = score.get("ci95_high")
                writer.writerow(row)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="artifacts/phase8/full-validation/benchmark_validation.json")
    parser.add_argument("--predictions", default="artifacts/phase8/full-validation/per_query_validation.csv")
    parser.add_argument("--data-audit", default="data/processed/data_audit.json")
    parser.add_argument("--validation-data", default="data/processed/validation.jsonl")
    parser.add_argument("--out", default="src/citewise/research_summary.json")
    parser.add_argument("--comparison-csv", default="artifacts/phase9/model_comparison.csv")
    args = parser.parse_args()
    bundle = build_summary(args.benchmark, args.predictions, args.data_audit, args.out,
                           validation_path=args.validation_data, comparison_csv_path=args.comparison_csv)
    print(f"Created validation-only UI summary: {bundle['evaluation']['question_count']} questions, "
          f"{len(bundle['evaluation']['models'])} model states")


if __name__ == "__main__":
    main()
