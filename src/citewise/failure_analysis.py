"""Export inspectable validation cases; observable flags are not causal labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .dataset import load_papers
from .research_benchmark import read_checkpoint, write_csv
from .retrieval import MODEL_IDS, MODEL_SPECS, tokens


def analyze(out_dir, data_dir="data/processed"):
    root = Path(out_dir)
    evidence = json.loads((root / "benchmark_validation.json").read_text(encoding="utf-8"))
    papers = {p["id"]: p for p in load_papers(Path(data_dir) / "validation.jsonl")}
    records = {}
    lengths = {}
    for summary in evidence["summaries"]:
        name = summary["model"]
        result = read_checkpoint(root / "checkpoints" / name / "model.json")
        records[name] = {(r["paper_id"], r["question_id"]): r for r in result["rows"]}
        lengths[name] = {pid: count for p in result["papers"] for pid, count in p["passage_token_lengths"].items()}
    excluded = [{"paper_id": p["id"], "paper_title": p["title"], "question_id": q["id"], "question": q["text"],
                 "status": q["status"], "annotated_evidence_items": q.get("annotated_evidence_items", 0),
                 "unsupported_figure_items": q.get("unsupported_figure_items", 0)}
                for p in papers.values() for q in p["questions"] if q["status"] != "scorable"]
    write_csv(root / "excluded_validation_questions.csv", excluded)
    cases, comparison_counts = [], {}
    baseline = records["bm25"]
    for name in records:
        if name not in MODEL_IDS:
            continue
        counts = Counter()
        for key, neural in records[name].items():
            lexical = baseline[key]
            delta = neural["recall_at_5"] - lexical["recall_at_5"]
            direction = "neural_better" if delta > 0 else ("bm25_better" if delta < 0 else "tie")
            counts[direction] += 1
            paper = papers[key[0]]
            by_id = {p["id"]: p for p in paper["paragraphs"]}
            gold = [by_id[pid] for pid in neural["gold_ids"].split("|")]
            qtokens = set(tokens(neural["question"]))
            coverage = max(len(qtokens & set(tokens(p["text"]))) / max(len(qtokens), 1) for p in gold)
            truncated = [p["id"] for p in gold if lengths[name].get(p["id"], 0) > MODEL_SPECS[name]["max_seq_length"]]
            counts["gold_paragraph_exceeds_token_cap"] += bool(truncated)
            counts["low_lexical_overlap_candidate_review"] += coverage < .25
            counts["short_query_candidate_review"] += len(qtokens) <= 5
            if direction == "tie":
                continue
            cases.append({"model": name, "paper_id": key[0], "question_id": key[1],
                          "paper_title": paper["title"], "question": neural["question"],
                          "direction": direction, "recall_delta_neural_minus_bm25": delta,
                          "bm25_recall": lexical["recall_at_5"], "neural_recall": neural["recall_at_5"],
                          "max_query_gold_token_coverage": coverage,
                          "query_token_count": len(qtokens),
                          "gold_paragraph_exceeds_token_cap": bool(truncated),
                          "truncated_gold_ids": truncated,
                          "gold_passages": gold,
                          "bm25_passages": [by_id[pid] for pid in lexical["retrieved_ids"].split("|") if pid],
                          "neural_passages": [by_id[pid] for pid in neural["retrieved_ids"].split("|") if pid],
                          "interpretation": "Requires qualitative review; lexical coverage and truncation are observable flags only."})
        comparison_counts[name] = dict(counts)
    (root / "failure_analysis.json").write_text(json.dumps({"comparisons": comparison_counts, "cases": cases,
            "review_flag_definitions": {"low_lexical_overlap_candidate_review": "maximum query/gold unigram coverage below 0.25; possible terminology mismatch or paraphrase, not a cause label", "short_query_candidate_review": "six or fewer unique tokens; ambiguous intent is a human review question, not an automated finding", "gold_paragraph_exceeds_token_cap": "recorded input token count exceeds this encoder's configured maximum; decisive evidence position within the paragraph is unknown"},
            "excluded_counts": dict(Counter(q["status"] for q in excluded)), "test_split_accessed": False},
            indent=2, ensure_ascii=False), encoding="utf-8")
    flat = [{k: v for k,v in c.items() if k not in ("gold_passages", "bm25_passages", "neural_passages", "truncated_gold_ids")} for c in cases]
    if flat:
        write_csv(root / "failure_review_queue.csv", flat)
    lines = ["# Validation failure review", "", "These are measured disagreements, with real paper text. Mechanism labels require qualitative review.", ""]
    for name in comparison_counts:
        lines.extend([f"## {name}", "", str(comparison_counts[name]), ""])
        for direction in ("bm25_better", "neural_better"):
            selected = sorted((c for c in cases if c["model"] == name and c["direction"] == direction),
                              key=lambda c: abs(c["recall_delta_neural_minus_bm25"]), reverse=True)[:3]
            for case in selected:
                lines += [f"### {direction}: {case['question_id']}", "", f"Paper: {case['paper_title']} ({case['paper_id']})",
                          f"Question: {case['question']}", f"Recall BM25={case['bm25_recall']:.3f}; {name}={case['neural_recall']:.3f}",
                          f"Gold beyond token cap: {case['gold_paragraph_exceeds_token_cap']}", "", "Gold evidence:", ""]
                lines += [f"> {p['text']}" for p in case["gold_passages"]]
                lines += ["", "Top BM25 candidate:", "", "> " + case["bm25_passages"][0]["text"],
                          "", "Top neural candidate:", "", "> " + case["neural_passages"][0]["text"], ""]
    (root / "FAILURE_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison_counts, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/phase3/full-validation-v1")
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    analyze(args.out_dir, args.data_dir)


if __name__ == "__main__":
    main()
