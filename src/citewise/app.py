"""Mobile-friendly API/UI; every result comes from an indexed real paper."""
import os
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .dataset import load_papers
from .retrieval import MODEL_IDS, MODEL_REVISIONS, MODEL_SPECS, rank

app = FastAPI(title="CiteWise", version="0.1.0")
DATA_DIR = Path(os.environ.get("CITEWISE_DATA_DIR", "data/processed"))
os.environ.setdefault("CITEWISE_EMBED_CACHE_DIR", os.environ.get("CITEWISE_CACHE_DIR", "data/cache/embeddings"))
DEFAULT_MODEL = os.environ.get("CITEWISE_DEFAULT_MODEL", "e5-base-v2")
INTERACTIVE_NEURAL_MODELS = ("minilm", "bge-base", "e5-base-v2", "gte-base")
MODEL_LABELS = {
    "bm25": "BM25 — lexical baseline",
    "supervised-logistic": "Supervised logistic — trained lexical baseline",
    "minilm": "MiniLM-L6-v2 — neural retriever",
    "bge-base": "BGE-base-en-v1.5 — instructed neural retriever",
    "e5-base-v2": "E5-base-v2 — neural retriever (query:/passage:)",
    "gte-base": "GTE-base — neural retriever",
}
MODEL_DESCRIPTIONS = {
    "minilm": "Uncached mean-pooled embeddings are truncated at 256 tokens.",
    "bge-base": "Uses the BGE search instruction, CLS pooling and 512-token truncation.",
    "e5-base-v2": "Uses query: and passage: prefixes, mean pooling and 512-token truncation.",
    "gte-base": "Uses plain query and passage text, mean pooling and 512-token truncation.",
}
UI = Path(__file__).with_name("index.html")
RESEARCH_SUMMARY = Path(__file__).with_name("research_summary.json")


@lru_cache(maxsize=1)
def model_states():
    data_available = (DATA_DIR / "validation.jsonl").is_file()
    data_reason = "" if data_available else "Prepared QASPER validation data is not staged."
    states = [{"id": "bm25", "label": MODEL_LABELS["bm25"], "available": data_available,
               "status": "ready" if data_available else "data_not_staged", "reason": data_reason}]
    ranker_path = Path(os.environ.get("CITEWISE_SUPERVISED_MODEL", "artifacts/supervised_ranker.joblib"))
    ranker_available = ranker_path.is_file() and data_available
    states.append({"id": "supervised-logistic", "label": MODEL_LABELS["supervised-logistic"],
                   "available": ranker_available,
                   "status": ("ready" if ranker_available else
                             "data_not_staged" if not data_available else "artifact_not_staged"),
                   "reason": ("" if ranker_available else data_reason or
                              "The trained ranker artifact is not staged on this server.")})
    for name in INTERACTIVE_NEURAL_MODELS:
        try:
            staged = os.environ.get(f"CITEWISE_MODEL_PATH_{name.upper().replace('-', '_')}")
            if staged:
                path = Path(staged).expanduser()
            else:
                from huggingface_hub.constants import HF_HUB_CACHE
                repo = MODEL_IDS[name].replace("/", "--")
                revision = MODEL_REVISIONS[name]
                path = Path(HF_HUB_CACHE) / f"models--{repo}" / "snapshots" / revision
            staged = ((path / "modules.json").is_file()
                      and (any(path.rglob("*.safetensors")) or any(path.rglob("pytorch_model*.bin"))))
        except (ImportError, OSError, RuntimeError, ValueError):
            staged = False
        available = staged and data_available
        states.append({"id": name, "label": MODEL_LABELS[name], "available": available,
                       "status": ("weights_staged" if available else
                                 "data_not_staged" if not data_available else "weights_not_staged"),
                       "reason": ("" if available else data_reason or
                                  "Pinned local weights are not staged; runtime downloads are disabled."),
                       "spec": MODEL_SPECS[name]})
    return states


@lru_cache(maxsize=3)
def paper_map(split):
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No prepared data at {path}. Run python -m citewise.prepare first.")
    return {paper["id"]: paper for paper in load_papers(path)}


class QueryPayload(BaseModel):
    split: str = Field(default="validation", pattern="^validation$")
    paper_id: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=2000)
    model: str = "bm25"
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/", response_class=HTMLResponse)
def index():
    return UI.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ready" if (DATA_DIR / "validation.jsonl").exists() else "needs_data",
            "source": "allenai/qasper"}


@app.get("/api/config")
def config():
    models = model_states()
    available = [m["id"] for m in models if m["available"]]
    default = DEFAULT_MODEL if DEFAULT_MODEL in available else "bm25"
    return {"default_model": default, "configured_default_model": DEFAULT_MODEL,
            "default_model_fallback": default != DEFAULT_MODEL,
            "available_models": available, "models": models}


@app.get("/api/research")
def research():
    try:
        return json.loads(RESEARCH_SUMMARY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Saved validation comparison is unavailable.") from exc


@app.get("/api/papers")
def papers(split: str = Query(pattern="^validation$", default="validation")):
    try:
        return [{"id": p["id"], "title": p["title"], "paragraphs": len(p["paragraphs"]),
                 "questions": len(p["questions"])} for p in paper_map(split).values()]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/questions/{paper_id}")
def questions(paper_id: str, split: str = Query(pattern="^validation$", default="validation")):
    try:
        p = paper_map(split).get(paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if p is None:
        raise HTTPException(404, detail="Paper not in selected split")
    return [{"id": q["id"], "text": q["text"]} for q in p["questions"]]


@app.post("/api/search")
def search(payload: QueryPayload):
    states = {m["id"]: m for m in model_states()}
    if payload.model not in states:
        raise HTTPException(400, detail="Unknown retrieval model")
    if not states[payload.model]["available"]:
        raise HTTPException(status_code=503, detail=f"Selected model is unavailable: {states[payload.model]['reason']}")
    try:
        paper = paper_map(payload.split).get(payload.paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if paper is None:
        raise HTTPException(404, detail="Paper not in selected split")
    try:
        ranked, ms = rank(payload.question, paper, payload.model, payload.top_k)
    except (ImportError, OSError, RuntimeError) as exc:
        raise HTTPException(503, detail="Selected retriever is unavailable on this server. Choose BM25 or another installed model.") from exc
    return {"title": paper["title"], "model": payload.model, "latency_ms": round(ms, 2),
            "results": ranked, "disclaimer": "Relevant evidence candidates, not verified generated answers."}
