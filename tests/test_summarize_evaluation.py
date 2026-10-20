import csv
import json

from citewise.summarize_evaluation import summarize


def test_summarize_completed_predictions_uses_paired_paper_clusters(tmp_path):
    evaluation = {"split": "validation", "summaries": [
        {"model": "bm25", "questions_evaluated": 4},
        {"model": "minilm", "questions_evaluated": 4}]}
    evaluation_path = tmp_path / "evaluation_validation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    predictions_path = tmp_path / "predictions_validation.csv"
    rows = [
        {"model": model, "paper_id": paper, "question_id": qid, "latency_ms": 1,
         "recall": score, "mrr": score, "ndcg": score}
        for paper, qid, a, b in (("p1", "q1", 1., 0.), ("p1", "q2", 1., 0.),
                                 ("p2", "q3", 1., 0.), ("p2", "q4", 1., 0.))
        for model, score in (("bm25", a), ("minilm", b))]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = summarize(evaluation_path, predictions_path, tmp_path / "summary.json", reps=100, seed=42)
    assert result["confidence_intervals"]["bm25"]["paper_clusters"] == 2
    assert result["confidence_intervals"]["bm25"]["metrics"]["recall_at_5"]["mean"] == 1.0
    assert result["paired_comparisons"][0]["mean_difference_a_minus_b"] == 1.0
    assert (tmp_path / "paired_comparisons_validation.csv").exists()
