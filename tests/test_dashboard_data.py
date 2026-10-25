import csv
import hashlib
import json

import pytest

from citewise.dashboard_data import _object_digest, _recall_at_1, build_summary
from citewise.evaluation import metrics_for_ranking
from citewise.research_benchmark import cluster_intervals


def test_recall_at_1_uses_gold_denominator_and_paper_grouping():
    rows = [
        {"paper_id": "p1", "gold_ids": "a|b", "retrieved_ids": "a|c"},
        {"paper_id": "p1", "gold_ids": "c", "retrieved_ids": "x|c"},
        {"paper_id": "p2", "gold_ids": "d", "retrieved_ids": "d|e"},
    ]
    score = _recall_at_1(rows, reps=100, seed=42)
    assert score["mean"] == pytest.approx(.5)
    assert score["paper_clusters"] == 2
    assert score["bootstrap_reps"] == 100


def test_dashboard_bundle_uses_validation_predictions_and_checks_same_queries(tmp_path):
    validation_path = tmp_path / "validation.jsonl"
    paper_records = [
        {"id": "p1", "title": "Fixture one", "paragraphs": [{"id": "a", "text": "A"},
                                                                     {"id": "b", "text": "B"}],
         "questions": [{"id": "q1", "text": "Q1", "status": "scorable", "gold_paragraph_ids": ["a"]}]},
        {"id": "p2", "title": "Fixture two", "paragraphs": [{"id": "c", "text": "C"},
                                                                     {"id": "d", "text": "D"}],
         "questions": [{"id": "q2", "text": "Q2", "status": "scorable", "gold_paragraph_ids": ["c"]}]},
    ]
    validation_path.write_text("".join(json.dumps(row) + "\n" for row in paper_records), encoding="utf-8")
    data_sha = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    query_plan = hashlib.sha256(json.dumps([("p1", "q1", ["a"]), ("p2", "q2", ["c"])],
                                            sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    shared = {"protocol": "validation-only", "data_sha256": data_sha,
              "query_plan_sha256": query_plan, "normalize_embeddings": True,
              "precision": "float32", "similarity": "dot", "batch_size": 8,
              "threads": 4, "device": "cpu", "blas_threads": {}, "limit": 2, "k": 5}
    configs = {name: {**shared, "model": name, "model_id": name, "model_revision": None,
                      "encoder": {}, "hardware_and_software": {"device": "cpu"}}
               for name in ("bm25", "minilm")}
    predictions = {
        "bm25": [("p1", "q1", "a", ["a", "b"]), ("p2", "q2", "c", ["d", "c"])],
        "minilm": [("p1", "q1", "a", ["b", "a"]), ("p2", "q2", "c", ["c", "d"])],
    }
    rows = []
    for model, entries in predictions.items():
        for paper_id, question_id, gold_ids, ranked in entries:
            scores = metrics_for_ranking(ranked, gold_ids.split("|"), 5)
            rows.append({"model": model, "paper_id": paper_id, "question_id": question_id,
                         "gold_ids": gold_ids, "retrieved_ids": "|".join(ranked),
                         **{f"{key}_at_5": value for key, value in scores.items()}})
    grouped = {name: [row for row in rows if row["model"] == name] for name in predictions}
    confidence_intervals = {name: cluster_intervals(group, reps=100, seed=42)
                            for name, group in grouped.items()}
    summary_rows = []
    for name, group in grouped.items():
        config_id = _object_digest(configs[name])
        summary_rows.append({"model": name, "questions_evaluated": 2,
                             "recall_at_k": sum(r["recall_at_5"] for r in group) / 2,
                             "mrr_at_k": sum(r["mrr_at_5"] for r in group) / 2,
                             "ndcg_at_k": sum(r["ndcg_at_5"] for r in group) / 2,
                             "p50_latency_ms": 1, "p95_latency_ms": 2,
                             "latency_basis": "cached query", "checkpoint_id": config_id,
                             "peak_memory": {"peak_rss_bytes": 100}})
    paired = []
    from citewise.benchmark import paired_bootstrap
    for metric in ("recall_at_5", "mrr_at_5", "ndcg_at_5"):
        paired.append(paired_bootstrap(rows, "bm25", "minilm", metric, reps=100, seed=42))
    benchmark = {
        "split": "validation", "protocol": "validation-only", "paper_count": 2,
        "evaluated_paper_count": 2, "question_count": 2, "test_split_accessed": False,
        "summaries": summary_rows,
        "confidence_intervals": confidence_intervals,
        "configurations": configs,
        "phases": {name: {"model_loading_ms": 0, "paragraph_embedding_ms_total": 0}
                   for name in ("bm25", "minilm")},
        "paired_comparisons": paired,
    }
    audit = {"source": "allenai/qasper", "revision": "rev", "license": "CC BY 4.0",
             "splits": {"validation": {"papers": 2, "paragraphs": 2, "questions": 2,
                                        "questions_scorable": 2}}}
    benchmark_path, audit_path, prediction_path = (tmp_path / name for name in
                                                    ("benchmark.json", "audit.json", "predictions.csv"))
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "summary.json"
    comparison_csv = tmp_path / "comparison.csv"
    result = build_summary(benchmark_path, prediction_path, audit_path, output, reps=100,
                           validation_path=validation_path, comparison_csv_path=comparison_csv)
    assert result["evaluation"]["split"] == "validation"
    assert result["evaluation"]["test_split_accessed"] is False
    assert result["evaluation"]["evaluated_paper_clusters"] == 2
    assert result["evaluation"]["models"][0]["scores"]["recall_at_1"]["mean"] == .5
    assert result["evaluation"]["models"][0]["paper_count"] == 2
    assert len(result["evaluation"]["neural_recall_at_1_comparisons"]) == 0
    assert comparison_csv.exists()
    assert [m["id"] for m in result["evaluation"]["models"]][-2:] == ["e5-base-v2", "gte-base"]
    assert all(m["status"] == "not_evaluated" for m in result["evaluation"]["models"][-2:])
    assert result["evaluation"]["models"][-2]["model_id"] == "intfloat/e5-base-v2"
    assert result["evaluation"]["models"][-2]["spec"]["query_prefix"] == "query: "
    assert output.exists()

    benchmark["split"] = "test"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    with pytest.raises(ValueError, match="only accepts validation"):
        build_summary(benchmark_path, prediction_path, audit_path, output, reps=100,
                      validation_path=validation_path, comparison_csv_path=comparison_csv)
    benchmark["split"] = "validation"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    rows[0]["retrieved_ids"] = "wrong-paper-id"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Retrieved evidence is not in its source paper"):
        build_summary(benchmark_path, prediction_path, audit_path, output, reps=100,
                      validation_path=validation_path, comparison_csv_path=comparison_csv)
