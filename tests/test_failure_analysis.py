import json

from citewise.failure_analysis import analyze
from citewise.research_benchmark import save_checkpoint


def test_failure_exports_preserve_source_and_flag_truncated_gold(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    paper = {"id": "p", "title": "Unit fixture", "paragraphs": [
        {"id": "gold", "text": "Special terminology evidence.", "section": None},
        {"id": "other", "text": "Different information.", "section": "Other"}], "questions": [
        {"id": "q1", "text": "Special terminology?", "status": "scorable", "gold_paragraph_ids": ["gold"]},
        {"id": "q2", "text": "Unknown?", "status": "unanswerable", "gold_paragraph_ids": []}]}
    (data / "validation.jsonl").write_text(json.dumps(paper) + "\n", encoding="utf-8")
    base = {"paper_id": "p", "question_id": "q1", "question": "Special terminology?", "gold_ids": "gold"}
    for name, hit, retrieved in (("bm25", 1., "gold|other"), ("minilm", 0., "other")):
        rows = [{**base, "model": name, "recall_at_5": hit, "retrieved_ids": retrieved}]
        save_checkpoint(tmp_path / "checkpoints" / name / "model.json",
                        {"rows": rows, "papers": [{"passage_token_lengths": {"gold": 300}}]})
    (tmp_path / "benchmark_validation.json").write_text(json.dumps({"summaries": [{"model": "bm25"}, {"model": "minilm"}]}))
    analyze(tmp_path, data)
    result = json.loads((tmp_path / "failure_analysis.json").read_text())
    assert result["cases"][0]["gold_passages"][0]["text"] == "Special terminology evidence."
    assert result["cases"][0]["gold_paragraph_exceeds_token_cap"]
    assert result["comparisons"]["minilm"]["bm25_better"] == 1
    assert result["excluded_counts"] == {"unanswerable": 1}
    assert result["comparisons"]["minilm"]["gold_paragraph_exceeds_token_cap"] == 1
    assert "not a cause label" in result["review_flag_definitions"]["low_lexical_overlap_candidate_review"]
