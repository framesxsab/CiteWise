"""Render reproducible comparison figures from saved QASPER benchmark files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MODEL_LABELS = {
    "bm25": "BM25",
    "supervised-logistic": "Supervised logistic",
    "minilm": "MiniLM-L6-v2",
    "bge-base": "BGE-base-en-v1.5",
    "e5-base-v2": "E5-base-v2",
    "gte-base": "GTE-base",
}
MODEL_COLORS = {
    "bm25": "#62798a", "supervised-logistic": "#9aa8b1",
    "minilm": "#4b8f7b", "bge-base": "#34786a",
    "e5-base-v2": "#174f43", "gte-base": "#79a997",
}
METRIC_LABELS = {
    "recall_at_1": "Recall@1", "recall_at_5": "Recall@5",
    "mrr_at_5": "MRR@5", "ndcg_at_5": "nDCG@5",
}


def scope_label(data, rows):
    count = rows[0]["questions_evaluated"]
    full_validation = data.get("coverage", {}).get("scorable")
    split = data.get("split", "unknown")
    if split == "validation" and (data.get("limit") is not None
                                  or (full_validation is not None and count < full_validation)):
        return f"QASPER validation PILOT | {count} scorable questions"
    return f"QASPER {split} | {count} scorable questions"


def _save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=300, bbox_inches="tight")


def _paired_for(model_a, model_b, metric, comparisons):
    for item in comparisons:
        if item["metric"] != metric:
            continue
        if (item["model_a"], item["model_b"]) == (model_a, model_b):
            return item["mean_difference_a_minus_b"], item["ci95_low"], item["ci95_high"]
        if (item["model_a"], item["model_b"]) == (model_b, model_a):
            return (-item["mean_difference_a_minus_b"], -item["ci95_high"], -item["ci95_low"])
    return None


def main():
    import matplotlib.pyplot as plt
    import numpy as np
    from .dashboard_data import _recall_at_1

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/phase8/full-validation/benchmark_validation.json")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--summary", default="src/citewise/research_summary.json")
    parser.add_argument("--out-dir", default="artifacts/phase9/figures")
    args = parser.parse_args()
    source = Path(args.input)
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["summaries"]
    if not rows or data.get("split") != "validation":
        raise ValueError("Phase 9 figures require saved validation results")
    counts = {row["questions_evaluated"] for row in rows}
    if len(counts) != 1:
        raise ValueError("Different question counts; refusing a misleading chart")
    title_scope = f"QASPER {data['split']} | {rows[0]['questions_evaluated']} questions | {data['evaluated_paper_count']} papers"
    folder = Path(args.out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    predictions = Path(args.predictions) if args.predictions else source.with_name("per_query_validation.csv")
    with predictions.open(encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    by_model = {model: [row for row in query_rows if row["model"] == model]
                for model in (row["model"] for row in rows)}
    if any(len(by_model[row["model"]]) != row["questions_evaluated"] for row in rows):
        raise ValueError("Per-query predictions do not match aggregate question counts")
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if summary["source_artifacts"]["benchmark"]["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Dashboard summary was not built from this benchmark artifact")
    model_states = {model["id"]: model for model in summary["evaluation"]["models"]}
    if set(model_states) != set(by_model):
        raise ValueError("Dashboard summary and prediction files contain different models")

    # Each saved ranking is enough to recompute Recall@1; no retrieval model is loaded.
    recall1 = {name: _recall_at_1(model_rows) for name, model_rows in by_model.items()}
    metric_data = {
        "recall_at_1": {name: recall1[name] for name in by_model},
        "recall_at_5": {name: data["confidence_intervals"][name]["metrics"]["recall_at_5"] for name in by_model},
        "mrr_at_5": {name: data["confidence_intervals"][name]["metrics"]["mrr_at_5"] for name in by_model},
        "ndcg_at_5": {name: data["confidence_intervals"][name]["metrics"]["ndcg_at_5"] for name in by_model},
    }
    means = {
        "recall_at_1": {name: recall1[name]["mean"] for name in by_model},
        "recall_at_5": {row["model"]: row["recall_at_k"] for row in rows},
        "mrr_at_5": {row["model"]: row["mrr_at_k"] for row in rows},
        "ndcg_at_5": {row["model"]: row["ndcg_at_k"] for row in rows},
    }

    order = list(by_model)
    for metric, label in METRIC_LABELS.items():
        ranked = sorted(order, key=lambda name: means[metric][name], reverse=True)
        values = np.array([means[metric][name] * 100 for name in ranked])
        intervals = [metric_data[metric][name] for name in ranked]
        errors = np.array([[max(0, (values[i] / 100 - item["ci95_low"]) * 100) for i, item in enumerate(intervals)],
                           [max(0, (item["ci95_high"] - values[i] / 100) * 100) for i, item in enumerate(intervals)]])
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        colors = [MODEL_COLORS[name] for name in ranked]
        ax.barh(range(len(ranked)), values, xerr=errors, color=colors, capsize=3,
                error_kw={"elinewidth": 1.1, "capthick": 1.1})
        ax.set_yticks(range(len(ranked)), [MODEL_LABELS.get(name, name) for name in ranked])
        ax.invert_yaxis()
        ax.set_xticks(range(0, 61, 10), [f"{tick}%" for tick in range(0, 61, 10)])
        ax.set_xlabel(f"Macro {label} (percent)")
        ax.set_title(f"{title_scope} | {label}, 95% paper-cluster CI", fontsize=11)
        ax.grid(axis="x", color="#d8e0e4", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 74)
        ax.set_xticks(range(0, 61, 10), [f"{tick}%" for tick in range(0, 61, 10)])
        for y, value in enumerate(values):
            ax.text(66, y, f"{value:.2f}%", va="center", fontsize=8)
        _save(fig, folder, f"{metric}.png")
        plt.close(fig)

    ranked = sorted(order, key=lambda name: means["recall_at_5"][name], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, metric, label in zip(axes, ("recall_at_1", "recall_at_5"), ("Recall@1", "Recall@5")):
        values = np.array([means[metric][name] * 100 for name in ranked])
        intervals = [metric_data[metric][name] for name in ranked]
        errors = np.array([[max(0, (values[i] / 100 - item["ci95_low"]) * 100) for i, item in enumerate(intervals)],
                           [max(0, (item["ci95_high"] - values[i] / 100) * 100) for i, item in enumerate(intervals)]])
        ax.barh(range(len(ranked)), values, xerr=errors, color=[MODEL_COLORS[name] for name in ranked],
                capsize=3, error_kw={"elinewidth": 1.1, "capthick": 1.1})
        ax.set_yticks(range(len(ranked)), [MODEL_LABELS.get(name, name) for name in ranked])
        ax.invert_yaxis()
        ax.set_xlim(0, 74)
        ax.set_xticks(range(0, 61, 15), [f"{tick}%" for tick in range(0, 61, 15)])
        ax.set_xlabel(f"Macro {label} (percent)")
        ax.set_title(label)
        ax.grid(axis="x", color="#d8e0e4", linewidth=.7)
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            ax.text(66, index, f"{value:.1f}%", va="center", fontsize=8)
    fig.suptitle(f"{title_scope} | retrieval recall and 95% paper-cluster CIs")
    _save(fig, folder, "recall_at_1_5.png")
    plt.close(fig)

    # Grouped percentiles retain their units and use a declared log axis to show both baselines and encoders.
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    y = np.arange(len(order))
    height = .34
    p50 = [next(item["p50_latency_ms"] for item in rows if item["model"] == name) for name in order]
    p95 = [next(item["p95_latency_ms"] for item in rows if item["model"] == name) for name in order]
    ax.barh(y - height / 2, p50, height, color=[MODEL_COLORS[name] for name in order], label="Warm P50")
    ax.barh(y + height / 2, p95, height, color=[MODEL_COLORS[name] for name in order], alpha=.52, label="Warm P95")
    ax.set_xscale("log")
    ax.set_yticks(y, [MODEL_LABELS.get(name, name) for name in order])
    ax.invert_yaxis()
    ax.set_xticks([2, 5, 10, 20, 50, 100, 150], ["2", "5", "10", "20", "50", "100", "150"])
    ax.set_xlabel("Warm query retrieval latency (milliseconds, log scale)")
    ax.set_title(f"{title_scope} | model and passage vectors cached")
    ax.grid(axis="x", color="#d8e0e4", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2)
    for index, (left_value, right_value) in enumerate(zip(p50, p95)):
        ax.annotate(f"{left_value:.1f}", (left_value, y[index] - height / 2),
                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
        ax.annotate(f"{right_value:.1f}", (right_value, y[index] + height / 2),
                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
    _save(fig, folder, "warm_latency_percentiles.png")
    plt.close(fig)

    # Quality versus warm latency uses a log x-axis; the y-axis remains zero-based.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name in order:
        state = model_states[name]
        latency = state.get("latency_ms", {}).get("p50")
        if latency is None or latency <= 0:
            continue
        score = state["scores"]["recall_at_5"]
        color = MODEL_COLORS[name]
        family = "Neural encoder" if state["family"] == "neural retriever" else "Baseline"
        marker = "o" if family == "Neural encoder" else "s"
        ax.errorbar(latency, score["mean"] * 100,
                    yerr=[[max(0, score["mean"] - score["ci95_low"]) * 100],
                          [max(0, score["ci95_high"] - score["mean"]) * 100]],
                    fmt=marker, color=color, ecolor=color, capsize=3, markersize=6)
        ax.annotate(MODEL_LABELS.get(name, name), (latency, score["mean"] * 100),
                    xytext=(6, 5), textcoords="offset points", fontsize=8, color="#20302d")
    ax.set_xscale("log")
    ax.set_xlim(1.8, 160)
    ax.set_ylim(0, 65)
    ax.set_xticks([2, 5, 10, 25, 50, 100], ["2", "5", "10", "25", "50", "100"])
    ax.set_xlabel("Warm query P50 latency (ms, log scale)")
    ax.set_ylabel("Recall@5 (percent; paper-cluster 95% CI)")
    ax.set_title(f"{title_scope} | retrieval quality versus warm latency")
    ax.grid(color="#d8e0e4", linewidth=.7)
    ax.set_axisbelow(True)
    _save(fig, folder, "quality_vs_warm_latency.png")
    plt.close(fig)

    # Show paired E5 differences only, as this directly answers which neural point-estimate leader is separated.
    paired = summary["evaluation"].get("paired_comparisons", [])
    paired_r1 = summary["evaluation"].get("neural_recall_at_1_comparisons", [])
    comparison_points = []
    competitors = [name for name in ("minilm", "bge-base", "gte-base") if name in by_model]
    for metric in ("recall_at_1", "recall_at_5", "mrr_at_5", "ndcg_at_5"):
        source = paired_r1 if metric == "recall_at_1" else paired
        for name in competitors:
            point = _paired_for("e5-base-v2", name, metric, source)
            if point is not None:
                comparison_points.append((metric, name, point))
    if comparison_points:
        metric_color = {"recall_at_1": "#315c79", "recall_at_5": "#397d70",
                        "mrr_at_5": "#8a7041", "ndcg_at_5": "#8a5c79"}
        fig, ax = plt.subplots(figsize=(8.4, 5.8))
        y = np.arange(len(comparison_points))
        for index, (metric, name, point) in enumerate(comparison_points):
            mean, low, high = [value * 100 for value in point]
            ax.errorbar(mean, index, xerr=[[max(0, mean - low)], [max(0, high - mean)]],
                        fmt="o", color=metric_color[metric], capsize=3)
        ax.axvline(0, color="#606a70", linewidth=1)
        ax.set_yticks(y, [f"E5 - {MODEL_LABELS[name]} | {METRIC_LABELS[metric]}"
                          for metric, name, _ in comparison_points])
        ax.invert_yaxis()
        ax.set_xlabel("Paired mean difference (percentage points); 95% paper-cluster CI")
        ax.set_title(f"{title_scope} | E5-base-v2 minus other neural models")
        ax.grid(axis="x", color="#d8e0e4", linewidth=.7)
        ax.set_axisbelow(True)
        _save(fig, folder, "paired_neural_differences.png")
        plt.close(fig)

    # Model loading, document encoding and warm query latency remain distinct phases.
    phases = data.get("phases", {})
    if phases:
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        loads = np.array([phases[name]["model_loading_ms"] / 1000 for name in order])
        encodes = np.array([phases[name]["paragraph_embedding_ms_total"] / 1000 for name in order])
        y = np.arange(len(order))
        ax.barh(y, loads, label="Model loading (s)", color="#607989")
        ax.barh(y, encodes, left=loads, label="Document encoding (s)", color="#34786a")
        ax.set_xscale("symlog", linthresh=.01)
        ax.set_yticks(y, [MODEL_LABELS.get(name, name) for name in order])
        ax.invert_yaxis()
        ax.set_xlabel("Separate setup phase duration (seconds; symlog scale)")
        ax.set_title(f"{title_scope} | 12,869 paragraphs in neural encoders")
        ax.grid(axis="x", color="#d8e0e4", linewidth=.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        _save(fig, folder, "model_loading_and_document_encoding.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for name in order:
        latencies = sorted(float(row["latency_ms"]) for row in by_model[name])
        ax.step(latencies, [(index + 1) / len(latencies) for index in range(len(latencies))],
                where="post", color=MODEL_COLORS[name], label=MODEL_LABELS.get(name, name))
    ax.set_xscale("log")
    ax.set_xlabel("Warm query retrieval latency (ms, log scale)")
    ax.set_ylabel("Cumulative fraction of validation queries")
    ax.set_title(f"{title_scope} | warm latency distribution")
    ax.grid(color="#d8e0e4", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2)
    _save(fig, folder, "warm_latency_distribution.png")
    plt.close(fig)
    print(f"Created validation-only figures from {len(query_rows)} saved predictions in {folder}")


if __name__ == "__main__":
    main()
