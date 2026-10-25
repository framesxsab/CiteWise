import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from citewise.retrieval import (MODEL_REVISIONS, MODEL_SPECS, embedded_paper, encoder_texts,
                                model_snapshot, pooling_matches)
from citewise import research_benchmark as benchmark


def test_document_and_query_instructions_are_distinct():
    assert encoder_texts("minilm", ["question"], query=True) == ["question"]
    assert encoder_texts("minilm", ["evidence"]) == ["evidence"]
    assert encoder_texts("e5-base-v2", ["evidence"]) == ["passage: evidence"]
    assert encoder_texts("e5-base-v2", ["question"], query=True) == ["query: question"]
    assert encoder_texts("bge-base", ["question"], query=True)[0].startswith("Represent this sentence")
    assert encoder_texts("bge-base", ["evidence"]) == ["evidence"]
    assert encoder_texts("gte-base", ["question"], query=True) == ["question"]
    assert encoder_texts("gte-base", ["evidence"]) == ["evidence"]
    assert (MODEL_SPECS["minilm"]["max_seq_length"], MODEL_SPECS["minilm"]["pooling"]) == (256, "mean")
    assert (MODEL_SPECS["bge-base"]["max_seq_length"], MODEL_SPECS["bge-base"]["pooling"]) == (512, "cls")
    assert (MODEL_SPECS["e5-base-v2"]["max_seq_length"], MODEL_SPECS["e5-base-v2"]["pooling"]) == (512, "mean")
    assert (MODEL_SPECS["gte-base"]["max_seq_length"], MODEL_SPECS["gte-base"]["pooling"]) == (512, "mean")
    assert encoder_texts("qwen3-0.6b", ["question"], query=True)[0].endswith("\nQuery:question")
    assert len(MODEL_REVISIONS["e5-base-v2"]) == 40
    assert len(MODEL_REVISIONS["gte-base"]) == 40


def test_neural_query_and_document_embedding_paths_normalize_vectors(monkeypatch):
    import numpy as np
    from citewise import retrieval

    paper = {"id": "p1", "paragraphs": [
        {"id": "p1:a", "section": "Methods", "text": "First passage"},
        {"id": "p1:b", "section": "Results", "text": "Second passage"}]}
    monkeypatch.setenv("CITEWISE_DISABLE_DISK_EMBED_CACHE", "1")
    for name in ("minilm", "bge-base", "e5-base-v2", "gte-base"):
        calls = []
        class FakeModel:
            def encode(self, texts, **kwargs):
                calls.append((list(texts), kwargs))
                return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))
        monkeypatch.setattr(retrieval, "get_model", lambda _: FakeModel())
        retrieval.embedded_paper.cache_clear()
        ranked, _ = retrieval.rank("What happened?", paper, name, top_k=2)
        assert len(ranked) == 2
        assert len(calls) == 2
        assert all(call[1]["normalize_embeddings"] is True for call in calls)
        assert calls[0][0] == encoder_texts(name, ["What happened?"], query=True)
        assert calls[1][0] == encoder_texts(name, ["First passage", "Second passage"])
    retrieval.embedded_paper.cache_clear()


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


def test_atomic_checkpoint_failure_preserves_existing_file_and_leaves_valid_temp(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    old = {"version": 1}
    new = {"version": 2}
    benchmark.save_checkpoint(path, old)
    original = benchmark.os.replace

    def deny_replace(source, destination):
        if Path(destination) == path:
            raise PermissionError(5, "simulated access denied", str(destination))
        return original(source, destination)

    monkeypatch.setattr(benchmark.os, "replace", deny_replace)
    with pytest.raises(PermissionError):
        benchmark.save_checkpoint(path, new)

    assert benchmark.read_checkpoint(path) == old
    temps = list(tmp_path.glob("manifest.json.*.tmp"))
    assert len(temps) == 1
    assert benchmark.read_checkpoint(temps[0]) == new


@pytest.mark.skipif(os.name != "nt", reason="Windows error codes and sharing semantics")
def test_atomic_checkpoint_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    import ctypes

    path = tmp_path / "manifest.json"
    attempts = 0
    original = benchmark.os.replace

    def transient_then_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ctypes.WinError(32)
        return original(source, destination)

    monkeypatch.setattr(benchmark.os, "replace", transient_then_replace)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    benchmark.save_checkpoint(path, {"version": 3})
    assert attempts == 3
    assert benchmark.read_checkpoint(path) == {"version": 3}


def test_worker_lock_file_is_not_a_stale_lock_after_owner_exits(tmp_path):
    lock = tmp_path / "worker.lock"
    lock.write_bytes(b"0")
    for _ in range(2):
        with benchmark.model_lock(lock):
            assert lock.exists()
    assert lock.exists() and lock.stat().st_size == 1


def test_resume_preserves_only_checkpoint_io_compatible_source_fingerprint(tmp_path, monkeypatch):
    old_source = "legacy-runner-source"
    previous = {
        "protocol": benchmark.PROTOCOL, "data_sha256": "data", "query_plan_sha256": "queries",
        "source_sha256": old_source, "model": "bm25", "model_id": "bm25", "model_revision": None,
        "license": None, "encoder": None, "normalize_embeddings": True, "precision": "float32",
        "similarity": "normalized dot product", "batch_size": 8, "threads": 4, "device": "cpu",
        "blas_threads": {"OPENBLAS_NUM_THREADS": "1"}, "limit": 888, "k": 5,
        "hardware_and_software": {"python": "test"},
    }
    previous_id = benchmark.digest(previous)
    checkpoint_root = tmp_path / "checkpoints"
    manifest_dir = checkpoint_root / "bm25"
    manifest_dir.mkdir(parents=True)
    benchmark.save_checkpoint(manifest_dir / "manifest.json", {
        "config_id": previous_id, "config": previous, "attempts": [], "status": "running"})
    monkeypatch.setattr(benchmark, "LEGACY_FULL_RUN_CONFIG_ID", previous_id)
    monkeypatch.setattr(benchmark, "LEGACY_SOURCE_SHA256", old_source)
    monkeypatch.setattr(benchmark, "_legacy_checkpoint_source_digest", lambda: old_source)
    requested = {**previous, "source_sha256": "new-checkpoint-writer-source",
                 "model": "e5-base-v2", "model_id": "intfloat/e5-base-v2",
                 "model_revision": "pinned", "encoder": {"query_prefix": "query: "}}

    resolved = benchmark._source_fingerprint_for_resume(requested, checkpoint_root)
    assert resolved["source_sha256"] == old_source
    assert requested["source_sha256"] == "new-checkpoint-writer-source"


def test_assembly_allows_only_known_checkpoint_io_source_migration(monkeypatch):
    monkeypatch.setattr(benchmark, "_legacy_checkpoint_source_digest", lambda: "legacy")
    monkeypatch.setattr(benchmark, "_source_digest", lambda: "current")
    monkeypatch.setattr(benchmark, "LEGACY_SOURCE_SHA256", "legacy")
    monkeypatch.setattr(benchmark, "PHASE8_FULL_RUN_SOURCE_SHA256", "current")
    base = {"data_sha256": "data", "query_plan_sha256": "queries", "protocol": benchmark.PROTOCOL,
            "hardware_and_software": {"cpu_count": 16, "cuda_probe_performed": False,
                                       "cuda_available": None}}
    results = [
        {"config": {**base, "model": "bm25", "source_sha256": "legacy"}},
        {"config": {**base, "model": "minilm", "source_sha256": "current",
                     "hardware_and_software": {"cpu_count": 16, "cuda_probe_performed": True,
                                                "cuda_available": False}}},
    ]
    result = benchmark._assembly_source_compatibility(results)
    assert result["status"] == "checkpoint_io_migration"
    results[1]["config"]["data_sha256"] = "different-data"
    with pytest.raises(ValueError, match="different datasets"):
        benchmark._assembly_source_compatibility(results)


def test_resume_recovers_durable_paper_when_manifest_replace_is_interrupted(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(benchmark, "system_metadata", lambda **_: {"unit_test": True})
    papers = [
        {"id": "p1", "title": "Fixture one", "paragraphs": [
            {"id": "p1:s0:p0", "text": "Adam optimizer", "section": "Method"},
            {"id": "p1:s0:p1", "text": "Other details", "section": "Results"}], "questions": [
            {"id": "q1", "text": "Which optimizer?", "status": "scorable", "gold_paragraph_ids": ["p1:s0:p0"]},
            {"id": "q2", "text": "What is unrelated?", "status": "scorable", "gold_paragraph_ids": ["p1:s0:p1"]}]},
        {"id": "p2", "title": "Fixture two", "paragraphs": [
            {"id": "p2:s0:p0", "text": "Another finding", "section": "Results"}], "questions": [
            {"id": "q3", "text": "What finding?", "status": "scorable", "gold_paragraph_ids": ["p2:s0:p0"]}]},
    ]
    data = tmp_path / "data"
    data.mkdir()
    (data / "validation.jsonl").write_text("".join(json.dumps(p) + "\n" for p in papers), encoding="utf-8")
    args = SimpleNamespace(data_dir=str(data), out_dir=str(tmp_path / "full-validation"),
                           batch_size=8, device="cpu", threads=4, limit=3)
    real_replace = benchmark.os.replace
    manifest_replacements = 0

    def interrupt_after_paper_checkpoint(source, destination):
        nonlocal manifest_replacements
        if Path(destination).name == "manifest.json":
            manifest_replacements += 1
            if manifest_replacements == 4:
                raise PermissionError(5, "simulated manifest sharing denial", str(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(benchmark.os, "replace", interrupt_after_paper_checkpoint)
    with pytest.raises(PermissionError, match="sharing denial"):
        benchmark.run_model(args, "bm25")
    monkeypatch.setattr(benchmark.os, "replace", real_replace)

    model_dir = tmp_path / "full-validation" / "checkpoints" / "bm25"
    checkpoint_files = list(model_dir.glob("*.json"))
    checkpoint_files = [p for p in checkpoint_files if p.name not in {"manifest.json", "manifest.tmp", "model.json"}]
    assert len(checkpoint_files) == 1
    first_paper = benchmark.read_checkpoint(checkpoint_files[0])
    assert len(first_paper["rows"]) == 2
    main = benchmark.read_checkpoint(model_dir / "manifest.json")
    temp_paths = list(model_dir.glob("manifest.json.*.tmp"))
    assert len(temp_paths) == 1
    temp_manifest = benchmark.read_checkpoint(temp_paths[0])
    assert temp_manifest["config_id"] == main["config_id"]
    assert temp_manifest["attempts"][-1]["completed_papers"] == 1

    resumed = benchmark.run_model(args, "bm25")
    keys = [(row["paper_id"], row["question_id"]) for row in resumed["rows"]]
    assert len(keys) == 3 and len(set(keys)) == 3
    assert len(resumed["rows"]) == 3
    assert len(resumed["attempts"]) == 2
    assert (model_dir / "manifest.json").exists()
    assert temp_paths[0].exists()
    assert benchmark.read_checkpoint(model_dir / "manifest.json")["status"] == "complete"
