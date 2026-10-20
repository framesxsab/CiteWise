"""Query-time paragraph retrieval; no fake predictions, generated labels or stub scores."""
from __future__ import annotations

import math
import hashlib
import json
import os
import re
import time
from collections import Counter
from functools import lru_cache

import numpy as np

MODEL_IDS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "e5-base-v2": "intfloat/e5-base-v2",
    "gte-base": "thenlper/gte-base",
}
MODEL_REVISIONS = {
    "minilm": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "bge-base": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "e5-base-v2": "1c644c92ad3ba1efdad3f1451a637716616a20e8",
    "gte-base": "f846641ce4fb595a946b68fe18389798a8da9a64",
}
HEAVY_MODEL_IDS = {"bge-m3", "qwen3-0.6b"}
MODEL_SPECS = {
    "minilm": {"max_seq_length": 256, "pooling": "mean", "query_prefix": "", "passage_prefix": ""},
    "bge-base": {"max_seq_length": 512, "pooling": "cls", "query_prefix": "Represent this sentence for searching relevant passages: ", "passage_prefix": ""},
    "bge-m3": {"max_seq_length": 8192, "pooling": "cls", "query_prefix": "", "passage_prefix": ""},
    "qwen3-0.6b": {"max_seq_length": 32768, "pooling": "lasttoken", "query_prefix": "Instruct: Given a question about a scientific paper, retrieve passages that provide evidence for answering it.\nQuery:", "passage_prefix": ""},
    "e5-base-v2": {"max_seq_length": 512, "pooling": "mean", "query_prefix": "query: ", "passage_prefix": "passage: "},
    "gte-base": {"max_seq_length": 512, "pooling": "mean", "query_prefix": "", "passage_prefix": ""},
}


def model_snapshot(short_name):
    staged = os.environ.get(f"CITEWISE_MODEL_PATH_{short_name.upper().replace('-', '_')}")
    if staged:
        from pathlib import Path
        path = Path(staged).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"Configured model path does not exist: {path}")
        return str(path.resolve())
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return snapshot_download(MODEL_IDS[short_name], revision=MODEL_REVISIONS.get(short_name), local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(f"{short_name} is optional and is not cached locally; prefetch separately or use the documented hosted-GPU workflow") from exc


def model_revision(short_name):
    if short_name in MODEL_REVISIONS:
        return MODEL_REVISIONS[short_name]
    return os.path.basename(model_snapshot(short_name))


def encoder_texts(name, texts, query=False):
    prefix = MODEL_SPECS[name]["query_prefix" if query else "passage_prefix"]
    return [prefix + text for text in texts]


def pooling_matches(config, mode):
    if "pooling_mode" in config:
        actual = config["pooling_mode"]
        return actual == mode or actual == [mode] or actual == (mode,)
    key = {"cls": "pooling_mode_cls_token", "mean": "pooling_mode_mean_tokens",
           "lasttoken": "pooling_mode_lasttoken"}[mode]
    return bool(config.get(key))


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def bm25_scores(query, passages, k1=1.5, b=0.75):
    """Standard BM25 with document-frequency IDF smoothed for small corpora."""
    docs = [tokens(s) for s in passages]
    n = len(docs)
    if not n:
        return np.array([], dtype=float)
    lengths = [len(d) for d in docs]
    avgdl = max(sum(lengths) / n, 1e-12)
    dfs = Counter(t for doc in docs for t in set(doc))
    q_terms = set(tokens(query))
    idf = {t: math.log(1 + (n - dfs.get(t, 0) + .5) / (dfs.get(t, 0) + .5)) for t in q_terms}
    out = []
    for doc, dl in zip(docs, lengths):
        freq = Counter(doc)
        norm = k1 * (1 - b + b * dl / avgdl)
        out.append(sum(idf[t] * (freq[t] * (k1 + 1) / (freq[t] + norm))
                       for t in q_terms if freq[t]))
    return np.array(out, dtype=float)


@lru_cache(maxsize=1)
def get_supervised_ranker():
    from joblib import load
    path = os.environ.get("CITEWISE_SUPERVISED_MODEL", "artifacts/supervised_ranker.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Train the supervised model first: {path}")
    return load(path)


@lru_cache(maxsize=1)
def get_model(short_name):
    if short_name not in MODEL_IDS:
        raise ValueError(f"Unknown model: {short_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_snapshot(short_name), trust_remote_code=False,
                                local_files_only=True, device=os.environ.get("CITEWISE_DEVICE"))
    model.max_seq_length = MODEL_SPECS[short_name]["max_seq_length"]
    pools = [m for m in model if type(m).__name__ == "Pooling"]
    if len(pools) != 1 or not pooling_matches(pools[0].get_config_dict(), MODEL_SPECS[short_name]["pooling"]):
        raise RuntimeError(f"Unexpected pooling configuration for {short_name}")
    model.float()
    return model


@lru_cache(maxsize=1)
def embedded_paper(short_name, paper_id, passage_texts):
    cache_dir = os.environ.get("CITEWISE_EMBED_CACHE_DIR")
    cache_path = None
    if cache_dir and not os.environ.get("CITEWISE_DISABLE_DISK_EMBED_CACHE"):
        fingerprint = hashlib.sha256(json.dumps(
            [short_name, model_revision(short_name), MODEL_SPECS[short_name],
             "normalized-float32-v1", os.environ.get("CITEWISE_BATCH_SIZE", "16"),
             os.environ.get("CITEWISE_DEVICE", "cpu"), paper_id, passage_texts],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        cache_path = os.path.join(cache_dir, fingerprint + ".npy")
        if os.path.isfile(cache_path):
            try:
                matrix = np.load(cache_path, allow_pickle=False)
                if matrix.ndim == 2 and matrix.shape[0] == len(passage_texts) and matrix.shape[1] > 0 and np.isfinite(matrix).all():
                    return matrix
            except (EOFError, OSError, ValueError):
                pass
    model = get_model(short_name)
    matrix = model.encode(encoder_texts(short_name, passage_texts), prompt="",
                          batch_size=int(os.environ.get("CITEWISE_BATCH_SIZE", "16")),
                          normalize_embeddings=True, show_progress_bar=False)
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        temp_path = cache_path + f".{os.getpid()}.{time.time_ns()}.tmp"
        with open(temp_path, "wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        os.replace(temp_path, cache_path)
    return matrix


def rank(query, paper, name="bm25", top_k=5):
    passages = paper["paragraphs"]
    if not passages:
        return [], 0.0
    start = time.perf_counter()
    texts = [p["text"] for p in passages]
    if name == "bm25":
        scores = bm25_scores(query, texts)
    elif name == "supervised-logistic":
        from .training import ranker_features
        scores = get_supervised_ranker().predict_proba(ranker_features(query, paper))[:,1]
    elif name in MODEL_IDS:
        model = get_model(name)
        qvector = model.encode(encoder_texts(name, [query], query=True), prompt="",
                               normalize_embeddings=True, show_progress_bar=False)[0]
        matrix = embedded_paper(name, paper["id"], tuple(texts))
        scores = matrix @ qvector
    else:
        raise ValueError(f"Unknown retriever: {name}")
    ranked = np.argsort(-scores, kind="stable")[:top_k]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return [dict(passages[int(i)], score=float(scores[int(i)]), rank=position + 1)
            for position, i in enumerate(ranked)], elapsed_ms
