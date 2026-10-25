import importlib
import json


def test_interactive_models_are_only_enabled_when_artifacts_are_staged(tmp_path, monkeypatch):
    app = importlib.import_module("citewise.app")
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    (tmp_path / "validation.jsonl").write_text(json.dumps(
        {"id": "p1", "title": "fixture", "paragraphs": [], "questions": []}) + "\n", encoding="utf-8")
    ranker = tmp_path / "missing-ranker.joblib"
    monkeypatch.setenv("CITEWISE_SUPERVISED_MODEL", str(ranker))

    staged = tmp_path / "minilm"
    staged.mkdir()
    (staged / "modules.json").write_text("[]", encoding="utf-8")
    (staged / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setenv("CITEWISE_MODEL_PATH_MINILM", str(staged))
    for name in ("BGE_BASE", "E5_BASE_V2", "GTE_BASE"):
        monkeypatch.setenv(f"CITEWISE_MODEL_PATH_{name}", str(tmp_path / name))

    app.model_states.cache_clear()
    try:
        states = {row["id"]: row for row in app.model_states()}
        assert states["bm25"]["available"] is True
        assert states["supervised-logistic"]["available"] is False
        assert states["minilm"]["available"] is True
        assert states["bge-base"]["available"] is False
        assert states["e5-base-v2"]["available"] is False
        assert states["gte-base"]["available"] is False
    finally:
        app.model_states.cache_clear()


def test_invalid_or_partial_snapshot_is_not_advertised(tmp_path, monkeypatch):
    app = importlib.import_module("citewise.app")
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    (tmp_path / "validation.jsonl").write_text(json.dumps(
        {"id": "p1", "title": "fixture", "paragraphs": [], "questions": []}) + "\n", encoding="utf-8")
    staged = tmp_path / "partial"
    staged.mkdir()
    (staged / "modules.json").write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setenv("CITEWISE_SUPERVISED_MODEL", str(tmp_path / "missing-ranker.joblib"))
    monkeypatch.setenv("CITEWISE_MODEL_PATH_MINILM", str(staged))
    for name in ("BGE_BASE", "E5_BASE_V2", "GTE_BASE"):
        monkeypatch.setenv(f"CITEWISE_MODEL_PATH_{name}", str(tmp_path / name))

    app.model_states.cache_clear()
    try:
        minilm = next(row for row in app.model_states() if row["id"] == "minilm")
        assert minilm["available"] is False
        assert minilm["status"] == "weights_not_staged"
    finally:
        app.model_states.cache_clear()


def test_unavailable_configured_default_falls_back_to_bm25(monkeypatch):
    app = importlib.import_module("citewise.app")
    monkeypatch.setattr(app, "DEFAULT_MODEL", "e5-base-v2")
    monkeypatch.setattr(app, "model_states", lambda: [
        {"id": "bm25", "available": True, "label": "BM25"},
        {"id": "e5-base-v2", "available": False, "label": "E5"},
    ])
    config = app.config()
    assert config["default_model"] == "bm25"
    assert config["default_model_fallback"] is True


def test_e5_is_default_only_when_its_weights_are_available(monkeypatch):
    app = importlib.import_module("citewise.app")
    monkeypatch.setattr(app, "DEFAULT_MODEL", "e5-base-v2")
    monkeypatch.setattr(app, "model_states", lambda: [
        {"id": "bm25", "available": True, "label": "BM25"},
        {"id": "e5-base-v2", "available": True, "label": "E5"},
    ])
    config = app.config()
    assert config["default_model"] == "e5-base-v2"
    assert config["default_model_fallback"] is False


def test_models_are_not_advertised_without_validation_data(tmp_path, monkeypatch):
    app = importlib.import_module("citewise.app")
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    monkeypatch.setenv("CITEWISE_SUPERVISED_MODEL", str(tmp_path / "ranker.joblib"))
    app.model_states.cache_clear()
    try:
        states = app.model_states()
        assert states and all(not model["available"] for model in states)
        assert all(model["status"] == "data_not_staged" for model in states)
    finally:
        app.model_states.cache_clear()
