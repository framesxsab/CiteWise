"""Generate figures only from actual evaluation_{split}.json results."""
import argparse
import csv
import json
from pathlib import Path


def main():
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="artifacts/evaluation_validation.json")
    parser.add_argument("--out-dir",default="artifacts/figures")
    args=parser.parse_args()
    data=json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows=data["summaries"]
    if not rows:
        raise ValueError("No evaluation results to plot")
    counts={r["questions_evaluated"] for r in rows}
    if len(counts) != 1:
        raise ValueError("Different test sample counts; refusing a misleading chart")
    folder=Path(args.out_dir)
    folder.mkdir(parents=True,exist_ok=True)
    latency_label = ("Warm query P95 latency (milliseconds)" if all("latency_basis" in r for r in rows)
                     else "Observed P95 latency (ms; first-use effects may be included)")
    for metric, xlabel, filename in [
        ("ndcg_at_k","nDCG@5","ndcg_at_5.png"),
        ("recall_at_k","Recall@5","recall_at_5.png"),
        ("mrr_at_k","MRR@5","mrr_at_5.png"),
        ("p95_latency_ms",latency_label,"latency_p95.png")]:
        fig,ax=plt.subplots(figsize=(7,3.3))
        values = [r[metric] for r in rows]
        ci_key = {"recall_at_k": "recall_at_5", "mrr_at_k": "mrr_at_5", "ndcg_at_k": "ndcg_at_5"}.get(metric)
        errors = None
        if ci_key and data.get("confidence_intervals"):
            intervals = [data["confidence_intervals"][r["model"]]["metrics"][ci_key] for r in rows]
            errors = [[max(0, v - ci["ci95_low"]) for v, ci in zip(values, intervals)],
                      [max(0, ci["ci95_high"] - v) for v, ci in zip(values, intervals)]]
        ax.barh([r["model"] for r in rows], values, xerr=errors, capsize=3 if errors else 0)
        if metric == "p95_latency_ms":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Retriever")
        ax.set_title(f"QASPER {data['split']} | {next(iter(counts))} scorable questions")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(folder/filename,dpi=180)
        plt.close(fig)
    per_query = Path(args.input).with_name(f"per_query_{data['split']}.csv")
    if not per_query.exists():
        per_query = Path(args.input).with_name(f"predictions_{data['split']}.csv")
    if per_query.exists():
        with per_query.open(encoding="utf-8", newline="") as handle:
            query_rows = list(csv.DictReader(handle))
        fig, ax = plt.subplots(figsize=(8, 4))
        cached_queries = all("latency_basis" in row and "cached" in row["latency_basis"] for row in rows)
        latency_xlabel = ("Warm query retrieval latency (ms, log scale)" if cached_queries else
                          "Observed per-query retrieval latency; first-use effects may be included (ms, log scale)")
        for row in rows:
            latencies = sorted(float(q["latency_ms"]) for q in query_rows if q["model"] == row["model"])
            ax.step(latencies, [(i + 1) / len(latencies) for i in range(len(latencies))], label=row["model"])
        ax.set(xscale="log", xlabel=latency_xlabel, ylabel="Cumulative query fraction",
               title=f"QASPER {data['split']} | observed latency distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder/"latency_distribution.png", dpi=180)
        plt.close(fig)
    if data.get("paired_comparisons"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        for ax, key, title in zip(axes, ("recall_at_5", "mrr_at_5", "ndcg_at_5"), ("Recall@5", "MRR@5", "nDCG@5")):
            pairs = [p for p in data["paired_comparisons"] if p["metric"] == key]
            labels = [p["model_a"] + " − " + p["model_b"] for p in pairs]
            delta = [p["mean_difference_a_minus_b"] for p in pairs]
            errors = [[max(0, v-p["ci95_low"]) for p,v in zip(pairs,delta)],
                      [max(0, p["ci95_high"]-v) for p,v in zip(pairs,delta)]]
            ax.errorbar(delta, labels, xerr=errors, fmt="o", capsize=3)
            ax.axvline(0, color="grey", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Paired mean difference and 95% paper bootstrap CI")
        fig.tight_layout()
        fig.savefig(folder/"paired_differences.png", dpi=180)
        plt.close(fig)
    if data.get("phases"):
        fig, ax = plt.subplots(figsize=(8, 3.8))
        names = [r["model"] for r in rows]
        loading = [data["phases"][name]["model_loading_ms"] for name in names]
        embedding = [data["phases"][name]["paragraph_embedding_ms_total"] for name in names]
        y = list(range(len(names)))
        ax.barh(y, loading, label="Model loading")
        ax.barh(y, embedding, left=loading, label="Paragraph embedding")
        ax.set_yticks(y, names)
        ax.set_xlabel("Milliseconds (measured phases; log scale)")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_title(f"QASPER {data['split']} | setup and embedding phases")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder/"inference_phases.png", dpi=180)
        plt.close(fig)
    print(f"Created {len(rows)}-model charts in {folder}")

if __name__ == "__main__":
    main()
