import pytest
import json
import importlib
import os
from citewise.dataset import process_paper, sequence_records, normalize
from citewise.evaluation import metrics_for_ranking, evaluate
from citewise.retrieval import bm25_scores, get_supervised_ranker, rank


def toy_paper():
    """Unit-test-only fabricated sentences; never used as app/demo/evaluation data."""
    return {
        "id":"p1", "title":"Unit test", "abstract":"",
        "full_text":{"section_name":["Methods"],"paragraphs":[[
            "We extract features from measured temperature and voltage.",
            "The authors trained a neural network with Adam optimizer."]]},
        "qas":{"question":["Which optimizer was used?"], "question_id":["q1"],
               "answers":[{"answer":[{"unanswerable":False,
                  "evidence":["The authors trained a neural network with Adam optimizer."]}]}]},
    }


def test_sequence_record_shape():
    assert sequence_records({"a":[1,2],"b":[3,4]}) == [{"a":1,"b":3},{"a":2,"b":4}]
    with pytest.raises(ValueError): sequence_records({"a":[1],"b":[2,3]})


def test_whitespace_normalization_and_true_evidence_mapping():
    p=toy_paper()
    p["qas"]["answers"][0]["answer"][0]["evidence"]=["  The authors  trained a neural network with Adam optimizer.  "]
    res=process_paper(p,"test")
    q=res["questions"][0]
    assert len(q["gold_paragraph_ids"]) == 1
    assert q["status"] == "scorable"
    assert q["gold_paragraph_ids"][0] == res["paragraphs"][1]["id"]
    assert normalize("A   B") == "a b"


def test_bm25_prefers_exact_evidence():
    paper=process_paper(toy_paper(),"validation")
    out, latency=rank("Which Adam optimizer trained the model?",paper,"bm25")
    assert out[0]["id"] == paper["paragraphs"][1]["id"]
    assert latency >= 0


def test_scoring():
    metrics=metrics_for_ranking(["a","b","c"],{"b"},k=3)
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg"] < 1
    with pytest.raises(ValueError): metrics_for_ranking(["x"],set())


def test_evaluate_no_fabricated_benchmark():
    paper=process_paper(toy_paper(),"validation")
    summary, rows=evaluate([paper],"bm25")
    assert summary["questions_evaluated"] == 1
    assert len(rows)==1
    assert summary["recall_at_k"] == 1.0


def test_float_is_not_fake_text_evidence():
    row=toy_paper()
    row["qas"]["answers"][0]["answer"][0]["evidence"]=["FLOAT SELECTED: Table 1"]
    q=process_paper(row,"train")["questions"][0]
    assert q["status"] == "unmatched_or_no_text_evidence"
    assert q["unsupported_figure_items"] == 1


def test_training_feature_matrix_and_pairs():
    from citewise.training import ranker_features, make_training_pairs
    paper=process_paper(toy_paper(), "train")
    x=ranker_features("Which optimizer was used?",paper)
    assert x.shape == (2,6)
    X,y,stats=make_training_pairs([paper])
    assert X.shape == (2,6)
    assert sorted(y.tolist()) == [0,1]
    assert stats["positive_pairs"]==1


def test_null_section_heading_is_safe_for_features_and_ranking():
    row=toy_paper()
    row["full_text"]["section_name"]=[None]
    paper=process_paper(row,"validation")
    from citewise.training import ranker_features
    assert paper["paragraphs"][0]["section"] is None
    assert ranker_features("Which optimizer was used?",paper).shape == (2,6)
    ranked,_=rank("Which optimizer was used?",paper,"bm25")
    assert ranked[0]["id"] == paper["paragraphs"][1]["id"]


def test_missing_supervised_artifact_fails_clearly(tmp_path,monkeypatch):
    missing=tmp_path/"not-trained.joblib"
    monkeypatch.setenv("CITEWISE_SUPERVISED_MODEL",str(missing))
    get_supervised_ranker.cache_clear()
    with pytest.raises(FileNotFoundError,match="Train the supervised model first"):
        get_supervised_ranker()
    get_supervised_ranker.cache_clear()


def test_heavy_models_are_cache_only(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from citewise.retrieval import get_model
    calls=[]
    def cache_only(repo_id,revision=None,local_files_only=False):
        calls.append((repo_id,revision,local_files_only))
        raise LocalEntryNotFoundError("no local snapshot")
    monkeypatch.setattr(huggingface_hub,"snapshot_download",cache_only)
    get_model.cache_clear()
    with pytest.raises(RuntimeError,match="optional and is not cached locally"):
        get_model("bge-m3")
    assert calls and calls[0][2] is True
    get_model.cache_clear()


def test_paper_grouped_bootstrap_is_paired_and_deterministic():
    from citewise.benchmark import paired_bootstrap
    rows=[{"model":"a","paper_id":"p1","question_id":"q1","recall_at_5":1.0},
          {"model":"b","paper_id":"p1","question_id":"q1","recall_at_5":0.0},
          {"model":"a","paper_id":"p2","question_id":"q2","recall_at_5":0.5},
          {"model":"b","paper_id":"p2","question_id":"q2","recall_at_5":0.5}]
    result=paired_bootstrap(rows,"a","b","recall_at_5",reps=100,seed=7)
    assert result["mean_difference_a_minus_b"] == pytest.approx(.5)
    assert result["ci95_low"] <= .5 <= result["ci95_high"]
    assert result["paper_clusters"] == 2


def test_candidate_reranker_only_reorders_bm25_shortlist():
    import numpy as np
    from citewise.ranker_diagnostics import candidate_rankings
    class Scores:
        def __init__(self, values): self.values=values
        def predict_proba(self, _):
            values=self.values
            return np.asarray([[1-v,v] for v in values])
    features=[[3,0,0,0,0,0],[1,0,0,0,0,0],[2,0,0,0,0,0]]
    rankings=candidate_rankings(features,["a","b","c"],Scores([.1,.9,.2]),Scores([.9,.1,.8]),candidate_k=2)
    assert rankings["bm25"][:2] == ["a","c"]
    assert rankings["standardized-bm25-top-2"] == ["a","c"]
    assert "b" not in rankings["standardized-bm25-top-2"]


def test_benchmark_reports_process_memory_when_supported():
    from citewise.benchmark import memory_snapshot
    snapshot=memory_snapshot()
    assert "rss_bytes" in snapshot and "peak_rss_bytes" in snapshot
    if os.name == "nt":
        assert snapshot["rss_bytes"] > 0
        assert snapshot["peak_rss_bytes"] >= snapshot["rss_bytes"]


def test_api_rejects_invalid_requests_and_returns_real_ranked_evidence(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    app_module=importlib.import_module("citewise.app")
    paper=process_paper(toy_paper(),"validation")
    (tmp_path/"validation.jsonl").write_text(json.dumps(paper)+"\n",encoding="utf-8")
    monkeypatch.setattr(app_module,"DATA_DIR",tmp_path)
    app_module.paper_map.cache_clear()
    with TestClient(app_module.app) as client:
        assert client.post("/api/search",json={"split":"invalid","paper_id":"p1","question":"abc"}).status_code == 422
        assert client.post("/api/search",json={"paper_id":"p1","question":"valid question","model":"unknown"}).status_code == 400
        response=client.post("/api/search",json={"paper_id":"p1","question":"Which Adam optimizer trained the model?","model":"bm25","top_k":5})
        assert response.status_code == 200
        body=response.json()
        assert body["results"][0]["id"] == paper["paragraphs"][1]["id"]
        assert len(body["results"]) == 2
        states = [
            {"id": "bm25", "label": "BM25", "available": True, "status": "ready", "reason": ""},
            {"id": "supervised-logistic", "label": "Logistic", "available": True, "status": "ready", "reason": ""},
            {"id": "minilm", "label": "MiniLM", "available": True, "status": "weights_staged", "reason": ""},
            {"id": "bge-base", "label": "BGE-base", "available": True, "status": "weights_staged", "reason": ""},
            {"id": "e5-base-v2", "label": "E5", "available": False, "status": "weights_not_staged", "reason": "weights not staged"},
            {"id": "gte-base", "label": "GTE", "available": True, "status": "weights_staged", "reason": ""},
        ]
        monkeypatch.setattr(app_module, "model_states", lambda: states)
        config=client.get("/api/config").json()
        assert config["default_model"] == "bm25"
        assert config["configured_default_model"] == "e5-base-v2"
        assert config["default_model_fallback"] is True
        assert "e5-base-v2" not in config["available_models"]
        assert next(m for m in config["models"] if m["id"] == "e5-base-v2")["available"] is False
        unavailable=client.post("/api/search",json={"paper_id":"p1","question":"valid question","model":"e5-base-v2"})
        assert unavailable.status_code == 503
        assert "unavailable" in unavailable.json()["detail"]
        def unavailable(*args,**kwargs):
            raise RuntimeError("model unavailable")
        monkeypatch.setattr(app_module,"rank",unavailable)
        inference_failure=client.post("/api/search",json={"paper_id":"p1","question":"valid question","model":"gte-base"})
        assert inference_failure.status_code == 503
        assert "unavailable" in inference_failure.json()["detail"]
        assert client.post("/api/search",json={"split":"test","paper_id":"p1","question":"valid question"}).status_code == 422
        assert client.get("/api/papers?split=test").status_code == 422
        research=client.get("/api/research").json()
        assert research["evaluation"]["split"] == "validation"
        assert research["evaluation"]["question_count"] == 888
        assert research["evaluation"]["test_split_accessed"] is False
    app_module.paper_map.cache_clear()
