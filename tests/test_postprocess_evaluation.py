import csv
import json

from citewise.postprocess_evaluation import postprocess


def test_postprocess_adds_paper_clustered_intervals_from_predictions_only(tmp_path):
    evaluation_path = tmp_path / "evaluation_validation.json"
    evaluation_path.write_text(json.dumps({"split": "validation", "summaries": [
        {"model": "bm25", "questions_evaluated": 4},
        {"model": "minilm", "questions_evaluated": 4}]}), encoding="utf-8")
    predictions_path = tmp_path / "predictions_validation.csv"
    rows = [{"model": model, "paper_id": paper, "question_id": question,
             "latency_ms": 1, "recall": score, "mrr": score, "ndcg": score}
            for paper, question, bm, nn in (("p1", "q1", 1., 0.), ("p1", "q2", 1., 0.),
                                             ("p2", "q3", 1., 0.), ("p2", "q4", 1., 0.))
            for model, score in (("bm25", bm), ("minilm", nn))]
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = postprocess(evaluation_path, predictions_path, tmp_path / "with_ci.json", reps=100, seed=42)
    assert result["confidence_intervals"]["bm25"]["paper_clusters"] == 2
    assert result["confidence_intervals"]["bm25"]["metrics"]["recall_at_5"]["mean"] == 1.0
    assert result["paired_comparisons"][0]["mean_difference_a_minus_b"] == 1.0
    assert (tmp_path / "paired_comparisons_validation.csv").exists()
