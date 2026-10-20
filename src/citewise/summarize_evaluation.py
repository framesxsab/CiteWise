"""Add paper-cluster uncertainty to completed evaluation predictions only."""
import argparse
import csv
import json
from pathlib import Path

from .benchmark import paired_bootstrap
from .research_benchmark import METRICS, cluster_intervals, write_csv


def summarize(evaluation_path, predictions_path, output_path, reps=2000, seed=42):
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    if evaluation.get("split") not in {"validation", "test"}:
        raise ValueError("Expected a completed validation or test evaluation file")
    with Path(predictions_path).open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        row = dict(source)
        row.update({f"{metric}_at_5": float(source[metric]) for metric in ("recall", "mrr", "ndcg")})
        rows.append(row)
    models = [result["model"] for result in evaluation["summaries"]]
    if len(rows) != sum(result["questions_evaluated"] for result in evaluation["summaries"]):
        raise ValueError("Prediction row count does not match completed evaluation summaries")
    intervals = {model: cluster_intervals([row for row in rows if row["model"] == model], reps, seed)
                 for model in models}
    paired = [paired_bootstrap(rows, a, b, metric, reps, seed)
              for index, a in enumerate(models) for b in models[index + 1:] for metric in METRICS]
    evaluation["confidence_intervals"] = intervals
    evaluation["paired_comparisons"] = paired
    evaluation["bootstrap"] = {"unit": "paper", "replicates": reps, "seed": seed,
                                "source": "completed per-query predictions; no dataset reload"}
    output_path = Path(output_path)
    output_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    write_csv(output_path.with_name(f"paired_comparisons_{evaluation['split']}.csv"), paired)
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    result = summarize(args.input, args.predictions, args.output, args.bootstrap_reps, args.seed)
    print(json.dumps({"split": result["split"], "models": len(result["summaries"]),
                      "questions_per_model": [r["questions_evaluated"] for r in result["summaries"]],
                      "paper_clusters": [v["paper_clusters"] for v in result["confidence_intervals"].values()],
                      "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
