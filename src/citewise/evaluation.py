"""Gold-evidence evaluation on held-out QASPER questions, measured not fabricated."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from .dataset import load_papers
from .retrieval import MODEL_IDS, rank


def metrics_for_ranking(retrieved_ids, relevant_ids, k=5):
    """Binary relevance: a retrieved passage is correct if any annotator cited it."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Unscorable question: no mapped textual evidence")
    retrieved_ids = list(retrieved_ids)[:k]
    hits = [1 if p in relevant else 0 for p in retrieved_ids]
    recall = len(set(retrieved_ids) & relevant) / len(relevant)
    mrr = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {"recall": recall, "mrr": mrr, "ndcg": dcg / ideal if ideal else 0.0}


def evaluate(papers, model, limit=None, k=5):
    results = []
    coverage = Counter()
    for paper in papers:
        for question in paper["questions"]:
            coverage[question["status"]] += 1
            if question["status"] != "scorable":
                continue
            if limit is not None and len(results) >= limit:
                break
            matches, ms = rank(question["text"], paper, model, top_k=k)
            metrics = metrics_for_ranking([r["id"] for r in matches], question["gold_paragraph_ids"], k=k)
            results.append({"model": model, "paper_id": paper["id"], "question_id": question["id"],
                            "latency_ms": round(ms, 4), **metrics})
        if limit is not None and len(results) >= limit:
            break
    if not results:
        raise RuntimeError("No scorable questions found; review data audit and evidence mapping")
    latency = sorted(r["latency_ms"] for r in results)
    p95 = latency[min(len(latency)-1, math.ceil(.95 * len(latency)) - 1)]
    summary = {"model": model, "questions_evaluated": len(results), "k": k,
               "recall_at_k": statistics.mean(r["recall"] for r in results),
               "mrr_at_k": statistics.mean(r["mrr"] for r in results),
               "ndcg_at_k": statistics.mean(r["ndcg"] for r in results),
               "p50_latency_ms": statistics.median(latency),
               "p95_latency_ms": p95, "coverage_seen": dict(coverage),
               "latency_note": "Warm/cache effects and hardware should be reported separately"}
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Run QASPER evidence-retrieval evaluation")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--models", nargs="+", choices=["bm25", "supervised-logistic", *MODEL_IDS], default=["bm25", "minilm", "bge-base"])
    parser.add_argument("--limit", type=int, default=None, help="Number of scorable questions; omit for full split")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    papers = list(load_papers(Path(args.data_dir) / f"{args.split}.jsonl"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    results = []
    for model in args.models:
        summary, rows = evaluate(papers, model, limit=args.limit)
        print(model, json.dumps(summary, indent=2))
        summaries.append(summary)
        results.extend(rows)
    meta = {"dataset": "allenai/qasper", "split": args.split, "limit": args.limit,
            "comparison_valid_only_with_equal_test_query_counts": True, "summaries": summaries}
    (out / f"evaluation_{args.split}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if results:
        with (out / f"predictions_{args.split}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)


if __name__ == "__main__":
    main()
