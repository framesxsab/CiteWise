"""Train-only diagnostics and bounded validation experiments for the lexical ranker."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import load_papers
from .evaluation import metrics_for_ranking
from .retrieval import bm25_scores
from .training import make_training_pairs, ranker_features

FEATURE_NAMES = ["bm25", "query_coverage", "doc_overlap", "relative_length",
                 "abstract_section", "conclusion_section"]


def feature_distribution(rows):
    values = np.asarray(rows, dtype=float)
    if values.size == 0:
        return {name: {"count": 0} for name in FEATURE_NAMES}
    return {
        name: {"count": int(values.shape[0]),
               "mean": float(np.mean(values[:, i])), "std": float(np.std(values[:, i])),
               "p50": float(np.quantile(values[:, i], .50)),
               "p90": float(np.quantile(values[:, i], .90)),
               "p99": float(np.quantile(values[:, i], .99))}
        for i, name in enumerate(FEATURE_NAMES)
    }


def candidate_rankings(features, paragraph_ids, raw_model, standardized_model, candidate_k=100):
    """Return actual full-paper and BM25-shortlist rankings for one paper/question."""
    features = np.asarray(features, dtype=float)
    if not len(paragraph_ids):
        return {"bm25": [], "supervised-logistic": [], "standardized-logistic": [],
                f"standardized-bm25-top-{candidate_k}": []}
    bm25 = np.argsort(-features[:, 0], kind="stable")
    raw_scores = raw_model.predict_proba(features)[:, 1]
    scaled_scores = standardized_model.predict_proba(features)[:, 1]
    shortlist = bm25[:candidate_k]
    return {
        "bm25": [paragraph_ids[i] for i in bm25],
        "supervised-logistic": [paragraph_ids[i] for i in np.argsort(-raw_scores, kind="stable")],
        "standardized-logistic": [paragraph_ids[i] for i in np.argsort(-scaled_scores, kind="stable")],
        f"standardized-bm25-top-{candidate_k}": [paragraph_ids[i] for i in shortlist[np.argsort(-scaled_scores[shortlist], kind="stable")]],
    }


def _coefficients(model):
    estimator = model.named_steps["logisticregression"] if hasattr(model, "named_steps") else model
    return {"feature_names": FEATURE_NAMES, "coef": estimator.coef_[0].astype(float).tolist(),
            "intercept": float(estimator.intercept_[0]),
            "classes": estimator.classes_.astype(int).tolist()}


def _first_scorable(papers, limit):
    count = 0
    for paper in papers:
        for question in paper["questions"]:
            if question["status"] != "scorable":
                continue
            yield paper, question
            count += 1
            if count >= limit:
                return


def run_diagnostics(data_dir="data/processed", model_path="artifacts/supervised_ranker.joblib",
                    limit=40, candidate_k=100, out_dir="artifacts/phase2/ranker"):
    from joblib import load
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_papers = list(load_papers(Path(data_dir) / "train.jsonl"))
    X, y, pair_counts = make_training_pairs(train_papers)
    if X.size == 0 or len(set(y.tolist())) < 2:
        raise ValueError("Training data lacks positive and negative pairs")
    original = load(model_path)
    standardized = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=1500, class_weight="balanced", random_state=42))
    standardized.fit(X, y)

    candidate_diagnostics = {}
    train_pair_features = {str(label): feature_distribution(X[y == label]) for label in (0, 1)}
    for split in ("train", "validation"):
        full_rows, full_labels, sampled_rows, sampled_labels = [], [], [], []
        paragraph_counts, sampled_counts = [], []
        questions = list(_first_scorable(load_papers(Path(data_dir) / f"{split}.jsonl"), limit))
        for paper, question in questions:
            features = ranker_features(question["text"], paper)
            gold = set(question["gold_paragraph_ids"])
            ids = [p["id"] for p in paper["paragraphs"]]
            labels = [int(pid in gold) for pid in ids]
            full_rows.extend(features)
            full_labels.extend(labels)
            paragraph_counts.append(len(ids))
            if split == "train":
                positive = [i for i, hit in enumerate(labels) if hit]
                negative = [i for i, hit in enumerate(labels) if not hit]
                negative.sort(key=lambda i: (-features[i, 0], i))
                chosen = positive + negative[:6]
                sampled_rows.extend(features[chosen])
                sampled_labels.extend(labels[i] for i in chosen)
                sampled_counts.append(len(chosen))
        candidate_diagnostics[split] = {
            "questions": len(questions), "unique_papers": len({p["id"] for p, _ in questions}),
            "full_candidate_count": len(full_rows),
            "positive_candidates": int(sum(full_labels)),
            "positive_candidate_fraction": float(np.mean(full_labels)) if full_labels else 0.0,
            "paragraphs_per_question_mean": statistics.mean(paragraph_counts) if paragraph_counts else 0.0,
            "sampled_pairs_per_train_question_mean": statistics.mean(sampled_counts) if sampled_counts else None,
            "full_candidate_features": feature_distribution(full_rows),
            "full_candidate_features_by_label": {
                str(label): feature_distribution(np.asarray(full_rows)[np.asarray(full_labels) == label])
                for label in (0, 1)
            },
        }
        if split == "train":
            candidate_diagnostics[split]["sampled_pair_count"] = len(sampled_rows)
            candidate_diagnostics[split]["sampled_positive_fraction"] = float(np.mean(sampled_labels)) if sampled_labels else 0.0
            candidate_diagnostics[split]["sampled_pair_features"] = feature_distribution(sampled_rows)
            candidate_diagnostics[split]["sampled_pair_features_by_label"] = {
                str(label): feature_distribution(np.asarray(sampled_rows)[np.asarray(sampled_labels) == label])
                for label in (0, 1)
            }

    validation = list(_first_scorable(load_papers(Path(data_dir) / "validation.jsonl"), limit))
    names = ["bm25", "supervised-logistic", "standardized-logistic", f"standardized-bm25-top-{candidate_k}"]
    rows, failures, metrics_by_model = [], [], {name: [] for name in names}
    for paper, question in validation:
        features = ranker_features(question["text"], paper)
        ids = [p["id"] for p in paper["paragraphs"]]
        rankings = candidate_rankings(features, ids, original, standardized, candidate_k)
        gold = set(question["gold_paragraph_ids"])
        shortlist_ids = set(rankings["bm25"][:candidate_k])
        candidate_recall = len(shortlist_ids & gold) / len(gold)
        for name in names:
            metric = metrics_for_ranking(rankings[name], gold, k=5)
            metrics_by_model[name].append(metric)
            rows.append({"model": name, "paper_id": paper["id"], "question_id": question["id"],
                         "question": question["text"], "recall_at_5": metric["recall"],
                         "mrr_at_5": metric["mrr"], "ndcg_at_5": metric["ndcg"],
                         "bm25_top_candidate_recall": candidate_recall,
                         "gold_ids": "|".join(sorted(gold)), "top5_ids": "|".join(rankings[name][:5])})
            if metric["recall"] == 0:
                failures.append(rows[-1])

    summaries = {name: {"questions_evaluated": len(values),
                        "recall_at_5": statistics.mean(x["recall"] for x in values),
                        "mrr_at_5": statistics.mean(x["mrr"] for x in values),
                        "ndcg_at_5": statistics.mean(x["ndcg"] for x in values)}
                 for name, values in metrics_by_model.items()}
    from .benchmark import paired_bootstrap
    paired = [paired_bootstrap(rows, a, b, metric) for i, a in enumerate(names)
              for b in names[i + 1:] for metric in ("recall_at_5", "mrr_at_5", "ndcg_at_5")]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence = {"dataset": "allenai/qasper", "training_split": "train", "decision_split": "validation",
                "validation_limit": limit, "candidate_k": candidate_k,
                "pair_sampling": {**pair_counts, "sampled_positive_fraction": float(np.mean(y)),
                                  "negative_cap_per_question": 6,
                                  "class_weight": "balanced",
                                  "sampled_pair_features_by_label": train_pair_features},
                "candidate_diagnostics": candidate_diagnostics,
                "coefficients": {"existing_raw_artifact": _coefficients(original),
                                 "standardized_refit": _coefficients(standardized)},
                "validation_summaries": summaries,
                "paired_validation_comparisons": paired,
                "validation_note": "Bounded validation experiment only; no test data read."}
    (out / "ranker_diagnostics.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    for name, data in (("ranker_experiment_validation.csv", rows), ("ranker_failures_validation.csv", failures),
                       ("ranker_paired_comparisons_validation.csv", paired)):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]) if data else
                                    ["model", "paper_id", "question_id", "question", "recall_at_5", "mrr_at_5", "ndcg_at_5", "bm25_top_candidate_recall", "gold_ids", "top5_ids"])
            writer.writeheader()
            writer.writerows(data)
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Diagnose logistic ranker using train fit and bounded validation only")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model", default="artifacts/supervised_ranker.joblib")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--out-dir", default="artifacts/phase2/ranker")
    args = parser.parse_args()
    if args.limit <= 0 or args.candidate_k <= 0:
        parser.error("--limit and --candidate-k must be positive")
    print(json.dumps(run_diagnostics(args.data_dir, args.model, args.limit,
                                     args.candidate_k, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
