"""Mobile-friendly API/UI; every result comes from an indexed real paper."""
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .dataset import load_papers
from .retrieval import MODEL_IDS, rank

app = FastAPI(title="CiteWise", version="0.1.0")
DATA_DIR = Path(os.environ.get("CITEWISE_DATA_DIR", "data/processed"))
os.environ.setdefault("CITEWISE_EMBED_CACHE_DIR", os.environ.get("CITEWISE_CACHE_DIR", "data/cache/embeddings"))
DEFAULT_MODEL = os.environ.get("CITEWISE_DEFAULT_MODEL", "bge-base")
if DEFAULT_MODEL not in ["bm25", "supervised-logistic", *MODEL_IDS]:
    raise RuntimeError(f"Unknown CITEWISE_DEFAULT_MODEL: {DEFAULT_MODEL}")
UI = Path(__file__).with_name("index.html")


@lru_cache(maxsize=3)
def paper_map(split):
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No prepared data at {path}. Run python -m citewise.prepare first.")
    return {paper["id"]: paper for paper in load_papers(path)}


class QueryPayload(BaseModel):
    split: str = Field(default="validation", pattern="^(train|validation|test)$")
    paper_id: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=2000)
    model: str = DEFAULT_MODEL
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
    return {"default_model": DEFAULT_MODEL, "available_models": ["bm25", "supervised-logistic", *MODEL_IDS]}


@app.get("/api/papers")
def papers(split: str = Query(pattern="^(train|validation|test)$", default="validation")):
    try:
        return [{"id": p["id"], "title": p["title"], "paragraphs": len(p["paragraphs"]),
                 "questions": len(p["questions"])} for p in paper_map(split).values()]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/questions/{paper_id}")
def questions(paper_id: str, split: str = Query(pattern="^(train|validation|test)$", default="validation")):
    try:
        p = paper_map(split).get(paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if p is None:
        raise HTTPException(404, detail="Paper not in selected split")
    return [{"id": q["id"], "text": q["text"]} for q in p["questions"]]


@app.post("/api/search")
def search(payload: QueryPayload):
    if payload.model not in ["bm25", "supervised-logistic", *MODEL_IDS]:
        raise HTTPException(400, detail="Unknown retrieval model")
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
