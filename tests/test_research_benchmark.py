import json
from types import SimpleNamespace

import pytest

from citewise.retrieval import (MODEL_REVISIONS, embedded_paper, encoder_texts,
                                model_snapshot, pooling_matches)
from citewise import research_benchmark as benchmark


def test_document_and_query_instructions_are_distinct():
    assert encoder_texts("e5-base-v2", ["evidence"]) == ["passage: evidence"]
    assert encoder_texts("e5-base-v2", ["question"], query=True) == ["query: question"]
    assert encoder_texts("bge-base", ["question"], query=True)[0].startswith("Represent this sentence")
    assert encoder_texts("bge-base", ["evidence"]) == ["evidence"]
    assert encoder_texts("qwen3-0.6b", ["question"], query=True)[0].endswith("\nQuery:question")
    assert len(MODEL_REVISIONS["e5-base-v2"]) == 40
    assert len(MODEL_REVISIONS["gte-base"]) == 40


def test_staged_model_path_is_supported_without_hub_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("CITEWISE_MODEL_PATH_E5_BASE_V2", str(tmp_path))
    assert model_snapshot("e5-base-v2") == str(tmp_path.resolve())


def test_e5_snapshot_uses_immutable_revision_and_cache_only(monkeypatch):
    import huggingface_hub
    seen = {}
    def cached(repo_id, revision=None, local_files_only=False):
        seen.update(repo_id=repo_id, revision=revision, local_files_only=local_files_only)
        return "snapshot"
    monkeypatch.delenv("CITEWISE_MODEL_PATH_E5_BASE_V2", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    assert model_snapshot("e5-base-v2") == "snapshot"
    assert seen == {"repo_id": "intfloat/e5-base-v2",
                    "revision": "1c644c92ad3ba1efdad3f1451a637716616a20e8",
                    "local_files_only": True}


def test_document_embeddings_persist_between_process_cache_resets(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setenv("CITEWISE_EMBED_CACHE_DIR", str(tmp_path / "embeddings"))
    fake_model = SimpleNamespace(encode=lambda texts, **kwargs: np.ones((len(texts), 3), dtype=np.float32))
    monkeypatch.setattr("citewise.retrieval.get_model", lambda _: fake_model)
    embedded_paper.cache_clear()
    first = embedded_paper("e5-base-v2", "paper-1", ("paragraph",))
    embedded_paper.cache_clear()
    monkeypatch.setattr("citewise.retrieval.get_model", lambda _: pytest.fail("disk cache should avoid re-encoding"))
    second = embedded_paper("e5-base-v2", "paper-1", ("paragraph",))
    np.testing.assert_array_equal(first, second)
    assert list((tmp_path / "embeddings").glob("*.npy"))
    embedded_paper.cache_clear()


def test_precompute_reads_only_selected_validation_split(tmp_path, monkeypatch):
    from citewise import prepare_embeddings
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    paper = {"id": "p1", "paragraphs": [{"id": "p1:s0:p0", "text": "real paragraph"}], "questions": []}
    (data_dir / "validation.jsonl").write_text(json.dumps(paper) + "\n", encoding="utf-8")
    import numpy as np
    from citewise import retrieval
    fake_model = SimpleNamespace(encode=lambda texts, **kwargs: np.ones((len(texts), 2), dtype=np.float32))
    monkeypatch.setattr(prepare_embeddings, "get_model", lambda _: fake_model)
    monkeypatch.setattr(retrieval, "get_model", lambda _: fake_model)
    monkeypatch.setenv("CITEWISE_EMBED_CACHE_DIR", str(tmp_path / "cache"))
    report = prepare_embeddings.prepare("e5-base-v2", data_dir, "validation", tmp_path / "cache", tmp_path / "report.json")
    assert report["split"] == "validation"
    assert report["paper_count"] == report["embedded_paper_count"] == 1
    assert not (data_dir / "test.jsonl").exists()


def test_pooling_validation_accepts_legacy_and_current_library_configs():
    assert pooling_matches({"pooling_mode_mean_tokens": True}, "mean")
    assert pooling_matches({"pooling_mode_cls_token": True}, "cls")
    assert pooling_matches({"pooling_mode": "mean"}, "mean")
    assert pooling_matches({"pooling_mode": ["lasttoken"]}, "lasttoken")
    assert not pooling_matches({"pooling_mode": "cls"}, "mean")


def test_checkpoint_detects_corruption_and_does_not_overwrite_previous(tmp_path):
    path = tmp_path / "checkpoint.json"
    benchmark.save_checkpoint(path, {"rows": [1, 2]})
    assert benchmark.read_checkpoint(path) == {"rows": [1, 2]}
    record = json.loads(path.read_text())
    record["payload"]["rows"] = [3]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        benchmark.read_checkpoint(path)


def test_paper_cluster_interval_with_single_cluster():
    rows = [{"paper_id": "one", "recall_at_5": r, "mrr_at_5": r, "ndcg_at_5": r} for r in [0., 1.]]
    result = benchmark.cluster_intervals(rows, reps=30)
    assert result["paper_clusters"] == 1
    assert result["metrics"]["recall_at_5"] == {"mean": .5, "ci95_low": .5, "ci95_high": .5}


def test_worker_resumes_completed_papers_and_rejects_changed_data(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(benchmark, "system_metadata", lambda **k: {"unit_test": True})
    paper = {"id": "fixture", "title": "Unit fixture only", "paragraphs": [
        {"id": "a", "text": "Adam optimizer", "section": None},
        {"id": "b", "text": "Unrelated details", "section": "Other"}], "questions": [
        {"id": "q", "text": "Adam optimizer?", "status": "scorable", "gold_paragraph_ids": ["a"]}]}
    (tmp_path / "validation.jsonl").write_text(json.dumps(paper) + "\n", encoding="utf-8")
    args = SimpleNamespace(data_dir=str(tmp_path), out_dir=str(tmp_path / "evidence"),
                           batch_size=16, device="cpu", threads=4, limit=1)
    first = benchmark.run_model(args, "bm25")
    monkeypatch.setattr(benchmark, "rank", lambda *a, **k: pytest.fail("completed paper was rerun"))
    resumed = benchmark.run_model(args, "bm25")
    assert resumed["rows"] == first["rows"]
    assert len(resumed["attempts"]) == 1
    (tmp_path / "validation.jsonl").write_text(json.dumps(paper) + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Resume configuration changed"):
        benchmark.run_model(args, "bm25")
