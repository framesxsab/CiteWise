"""Small transparent supervised evidence ranker, fit on train papers ONLY."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import load_papers
from .retrieval import bm25_scores, tokens


def ranker_features(question, paper):
    """Six non-leaky lexical/structural features for each candidate paragraph."""
    paragraphs = paper["paragraphs"]
    if not paragraphs:
        return np.zeros((0, 6), dtype=float)
    query_tokens = set(tokens(question))
    qlen = max(len(query_tokens), 1)
    raw_bm25 = bm25_scores(question, [p["text"] for p in paragraphs])
    section_names = [(p.get("section") or "") for p in paragraphs]
    features = []
    for p, sec, bms in zip(paragraphs, section_names, raw_bm25):
        doc_tokens = tokens(p["text"])
        doc_set = set(doc_tokens)
        overlap = len(query_tokens & doc_set)
        features.append([float(bms), overlap / qlen,
                         overlap / max(len(doc_set), 1),
                         len(doc_tokens) / 300,
                         int("abstract" in sec.casefold()),
                         int("conclusion" in sec.casefold())])
    return np.asarray(features, dtype=float)


def make_training_pairs(papers, max_negs_per_question=6, max_questions=None):
    X, y = [], []
    counters = Counter()
    for paper in papers:
        for question in paper["questions"]:
            if question["status"] != "scorable":
                counters["excluded_unscorable"] += 1
                continue
            if max_questions is not None and counters["questions"] >= max_questions:
                return np.asarray(X), np.asarray(y), dict(counters)
            features = ranker_features(question["text"], paper)
            gold = set(question["gold_paragraph_ids"])
            pos = [i for i, p in enumerate(paper["paragraphs"]) if p["id"] in gold]
            neg = [i for i, p in enumerate(paper["paragraphs"]) if p["id"] not in gold]
            if not pos or not neg:
                counters["excluded_degenerate"] += 1
                continue
            # Hardest lexical negatives help test if ranking improves beyond token overlap.
            neg.sort(key=lambda idx: (-features[idx,0], idx))
            keep = pos + neg[:max_negs_per_question]
            for i in keep:
                X.append(features[i])
                y.append(int(i in pos))
            counters["questions"] += 1
            counters["positive_pairs"] += len(pos)
            counters["negative_pairs"] += len(keep) - len(pos)
    return np.asarray(X), np.asarray(y), dict(counters)


def train(data_dir, output, max_questions=None):
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    papers = load_papers(Path(data_dir) / "train.jsonl")
    X, y, stats = make_training_pairs(papers, max_questions=max_questions)
    if X.size == 0 or len(set(y.tolist())) < 2:
        raise ValueError("Training data lacks positive and negative pairs")
    model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    model.fit(X, y)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output)
    record = {"dataset": "allenai/qasper", "split": "train", "model": "supervised-logistic-ranker",
              "feature_names": ["bm25", "query_coverage", "doc_overlap", "relative_length", "abstract_section", "conclusion_section"],
              "fit_samples": len(y), "training_counts": stats, "artifact": str(output)}
    output.with_suffix(".metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="data/processed")
    ap.add_argument("--out",default="artifacts/supervised_ranker.joblib")
    ap.add_argument("--max-questions",type=int,default=None, help="Only for smoke testing; final uses full train split")
    args=ap.parse_args()
    print(json.dumps(train(args.data_dir,args.out,args.max_questions),indent=2))

if __name__ == "__main__":
    main()
