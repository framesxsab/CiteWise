"""Resumable validation benchmark. Each model runs in a separate process."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

from .benchmark import memory_snapshot, paired_bootstrap, system_metadata, _questions
from .dataset import load_papers
from .evaluation import metrics_for_ranking
from .retrieval import (MODEL_IDS, MODEL_REVISIONS, MODEL_SPECS, embedded_paper, encoder_texts,
                        get_model, get_supervised_ranker, model_snapshot, rank)

PROTOCOL = "qasper-paragraph-retrieval-v3-instructed"
METRICS = ("recall_at_5", "mrr_at_5", "ndcg_at_5")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"sha256": digest(payload), "payload": payload},
                               ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_checkpoint(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("sha256") != digest(record["payload"]):
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return record["payload"]


@contextmanager
def model_lock(path):
    with Path(path).open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def cluster_intervals(rows, reps=2000, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[row["paper_id"]].append(row)
    papers = sorted(groups)
    counts = np.array([len(groups[p]) for p in papers])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(papers), np.full(len(papers), 1 / len(papers)), size=reps)
    intervals = {}
    for metric in METRICS:
        sums = np.array([sum(r[metric] for r in groups[p]) for p in papers])
        samples = (weights @ sums) / (weights @ counts)
        low, high = np.quantile(samples, [.025, .975])
        intervals[metric] = {"mean": statistics.mean(r[metric] for r in rows),
                             "ci95_low": float(low), "ci95_high": float(high)}
    return {"paper_clusters": len(papers), "query_count": len(rows),
            "bootstrap_reps": reps, "seed": seed, "metrics": intervals}


def _source_digest():
    root = Path(__file__).parent
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("research_benchmark.py", "retrieval.py", "evaluation.py")})


def run_model(args, name):
    started = time.perf_counter()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CITEWISE_DISABLE_DISK_EMBED_CACHE"] = "1"
    os.environ["CITEWISE_BATCH_SIZE"] = str(args.batch_size)
    os.environ["CITEWISE_DEVICE"] = args.device
    if name in MODEL_IDS:
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        import sentence_transformers
    hardware = system_metadata(probe_cuda=name in MODEL_IDS)
    runtime_ms = (time.perf_counter() - started) * 1000
    data_path = Path(args.data_dir) / "validation.jsonl"
    papers = list(load_papers(data_path))
    _, scorable, queries = _questions(papers, args.limit)
    snapshot = Path(model_snapshot(name)) if name in MODEL_IDS else None
    config = {"protocol": PROTOCOL, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "query_plan_sha256": digest([(p["id"], q["id"], q["gold_paragraph_ids"]) for p, q in queries]),
              "source_sha256": _source_digest(), "model": name, "model_id": MODEL_IDS.get(name, name),
              "model_revision": MODEL_REVISIONS.get(name, snapshot.name) if snapshot else None,
              "license": {"e5-base-v2": "MIT", "gte-base": "MIT"}.get(name),
              "encoder": MODEL_SPECS.get(name), "normalize_embeddings": True,
              "precision": "float32", "similarity": "normalized dot product",
              "batch_size": args.batch_size, "threads": args.threads, "device": args.device,
              "blas_threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "limit": args.limit, "k": 5, "hardware_and_software": hardware}
    if name == "supervised-logistic":
        artifact = Path(os.environ.get("CITEWISE_SUPERVISED_MODEL", "artifacts/supervised_ranker.joblib"))
        config["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config_id = digest(config)
    folder = Path(args.out_dir) / "checkpoints" / name
    folder.mkdir(parents=True, exist_ok=True)
    with model_lock(folder / "worker.lock"):
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = read_checkpoint(manifest_path)
            if manifest["config_id"] != config_id:
                raise ValueError(f"Resume configuration changed for {name}; use a NEW output directory")
        else:
            manifest = {"config_id": config_id, "config": config, "attempts": [], "status": "running"}
        groups = defaultdict(list)
        for paper, question in queries:
            groups[paper["id"]].append((paper, question))
        completed = {}
        for paper_id, group in groups.items():
            path = folder / (hashlib.sha256(paper_id.encode()).hexdigest() + ".json")
            if path.exists():
                checkpoint = read_checkpoint(path)
                expected = [q["id"] for _, q in group]
                if checkpoint["config_id"] != config_id or [r["question_id"] for r in checkpoint["rows"]] != expected:
                    raise ValueError(f"Checkpoint query mismatch: {path}")
                completed[paper_id] = checkpoint
        if len(completed) < len(groups):
            attempt = {"id": uuid.uuid4().hex, "started_at": datetime.now(timezone.utc).isoformat(),
                       "runtime_initialization_ms": runtime_ms, "elapsed_seconds": 0.0,
                       "completed_papers": 0}
            manifest["attempts"].append(attempt)
            save_checkpoint(manifest_path, manifest)
            load_start = time.perf_counter()
            model = get_model(name) if name in MODEL_IDS else (get_supervised_ranker() if name == "supervised-logistic" else None)
            attempt["model_loading_ms"] = (time.perf_counter() - load_start) * 1000
            if name in MODEL_IDS:
                attempt["encoder_runtime"] = {"embedding_dimension": model.get_sentence_embedding_dimension(),
                                               "max_seq_length": model.max_seq_length,
                                               "dtype": str(next(model.parameters()).dtype)}
            attempt["memory_after_model_load"] = memory_snapshot()
            attempt["elapsed_seconds"] = time.perf_counter() - started
            save_checkpoint(manifest_path, manifest)
            for paper_id, group in groups.items():
                if paper_id in completed:
                    continue
                paper = group[0][0]
                paper_start = time.perf_counter()
                token_lengths = {}
                audit_ms = 0.0
                if model is not None and name in MODEL_IDS:
                    audit_start = time.perf_counter()
                    text = encoder_texts(name, [p["text"] for p in paper["paragraphs"]])
                    lengths = model.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
                    token_lengths = {p["id"]: len(ids) for p, ids in zip(paper["paragraphs"], lengths)}
                    audit_ms = (time.perf_counter() - audit_start) * 1000
                embed_start = time.perf_counter()
                if name in MODEL_IDS:
                    embedded_paper(name, paper_id, tuple(p["text"] for p in paper["paragraphs"]))
                embedding_ms = (time.perf_counter() - embed_start) * 1000 if name in MODEL_IDS else 0.0
                rows = []
                for _, question in group:
                    retrieved, ms = rank(question["text"], paper, name, top_k=5)
                    scores = metrics_for_ranking([p["id"] for p in retrieved], question["gold_paragraph_ids"])
                    rows.append({"model": name, "paper_id": paper_id, "question_id": question["id"],
                                 "question": question["text"], "latency_ms": ms,
                                 **{f"{m}_at_5": scores[m] for m in ("recall", "mrr", "ndcg")},
                                 "gold_ids": "|".join(question["gold_paragraph_ids"]),
                                 "retrieved_ids": "|".join(p["id"] for p in retrieved),
                                 "checkpoint_id": config_id})
                checkpoint = {"config_id": config_id, "attempt_id": attempt["id"], "paper_id": paper_id,
                              "paragraph_count": len(paper["paragraphs"]), "rows": rows,
                              "embedding_ms": embedding_ms, "token_audit_ms": audit_ms,
                              "passage_token_lengths": token_lengths, "memory": memory_snapshot(),
                              "wall_seconds": time.perf_counter() - paper_start}
                save_checkpoint(folder / (hashlib.sha256(paper_id.encode()).hexdigest() + ".json"), checkpoint)
                completed[paper_id] = checkpoint
                attempt["completed_papers"] += 1
                attempt["elapsed_seconds"] = time.perf_counter() - started
                attempt["memory_peak"] = memory_snapshot()
                save_checkpoint(manifest_path, manifest)
                print(f"{name}: {len(completed)}/{len(groups)} papers; {sum(len(c['rows']) for c in completed.values())}/{len(queries)} queries", flush=True)
        manifest["status"] = "complete"
        if manifest["attempts"]:
            manifest["attempts"][-1].setdefault("finished_at", datetime.now(timezone.utc).isoformat())
        save_checkpoint(manifest_path, manifest)
        rows = [row for paper_id in groups for row in completed[paper_id]["rows"]]
        result = {"config_id": config_id, "config": config, "attempts": manifest["attempts"], "rows": rows,
                  "papers": list(completed.values()), "total_execution_seconds": sum(a["elapsed_seconds"] for a in manifest["attempts"]),
                  "timing_note": "Recorded attempt wall time includes initialization, audit, encoding, retrieval and checkpoint writes. Hard-kill time after the last checkpoint is unobservable.",
                  "process_memory_peak": max((a.get("memory_peak", a.get("memory_after_model_load", {})) for a in manifest["attempts"]),
                                             key=lambda m: m.get("peak_rss_bytes") or 0, default=memory_snapshot())}
        save_checkpoint(folder / "model.json", result)
        return result


def write_csv(path, rows, fields=None):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assemble(args):
    results = [read_checkpoint(Path(args.out_dir) / "checkpoints" / name / "model.json") for name in args.models]
    signatures = {(r["config"]["data_sha256"], r["config"]["query_plan_sha256"], r["config"]["source_sha256"], r["config"]["protocol"]) for r in results}
    if len(signatures) != 1:
        raise ValueError("Refusing to compare different datasets, question plans or protocols")
    rows = [row for result in results for row in result["rows"]]
    summaries, phases, intervals = [], {}, {}
    for name, result in zip(args.models, results):
        queries = result["rows"]
        latencies = [r["latency_ms"] for r in queries]
        intervals[name] = cluster_intervals(queries, args.bootstrap_reps)
        summaries.append({"model": name, "questions_evaluated": len(queries), "k": 5,
                          **{k: statistics.mean(r[v] for r in queries) for k, v in
                             (("recall_at_k", "recall_at_5"), ("mrr_at_k", "mrr_at_5"), ("ndcg_at_k", "ndcg_at_5"))},
                          "p50_latency_ms": statistics.median(latencies),
                          "p95_latency_ms": sorted(latencies)[int(np.ceil(.95 * len(latencies))) - 1],
                          "p99_latency_ms": sorted(latencies)[int(np.ceil(.99 * len(latencies))) - 1],
                          "latency_basis": "model and document encodings cached; query encoding, scoring and sorting",
                          "total_execution_seconds": result["total_execution_seconds"],
                          "peak_memory": result["process_memory_peak"], "checkpoint_id": result["config_id"]})
        phases[name] = {"model_loading_ms": sum(a.get("model_loading_ms", 0) for a in result["attempts"]),
                        "runtime_initialization_ms": sum(a["runtime_initialization_ms"] for a in result["attempts"]),
                        "paragraph_embedding_ms_total": sum(p["embedding_ms"] for p in result["papers"]),
                        "token_audit_ms": sum(p["token_audit_ms"] for p in result["papers"]),
                        "papers_embedded": len(result["papers"]) if name in MODEL_IDS else 0,
                        "paragraphs_embedded": sum(p["paragraph_count"] for p in result["papers"]) if name in MODEL_IDS else 0,
                        "attempt_count": len(result["attempts"])}
    pairs = [paired_bootstrap(rows, a, b, metric, args.bootstrap_reps)
             for i, a in enumerate(args.models) for b in args.models[i+1:] for metric in METRICS]
    papers = list(load_papers(Path(args.data_dir) / "validation.jsonl"))
    evidence = {"dataset": "allenai/qasper", "split": "validation", "protocol": PROTOCOL,
                "paper_count": len(papers), "evaluated_paper_count": len({r["paper_id"] for r in rows}),
                "question_count": sum(len(p["questions"]) for p in papers),
                "coverage": dict(Counter(q["status"] for p in papers for q in p["questions"])),
                "summaries": summaries, "phases": phases, "confidence_intervals": intervals,
                "paired_comparisons": pairs, "configurations": {r["config"]["model"]: r["config"] for r in results},
                "test_split_accessed": False, "bootstrap": {"unit": "paper", "replicates": args.bootstrap_reps, "seed": 42}}
    out = Path(args.out_dir)
    save_checkpoint(out / "run_evidence.json", evidence)
    (out / "benchmark_validation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    write_csv(out / "per_query_validation.csv", rows)
    write_csv(out / "failure_cases_validation.csv", [r for r in rows if r["recall_at_5"] == 0], list(rows[0]))
    if pairs:
        write_csv(out / "paired_comparisons_validation.csv", pairs)
    print(json.dumps({"summaries": summaries, "output": str(out)}, indent=2), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["bm25", "supervised-logistic", *MODEL_IDS], default=["bm25", "minilm", "bge-base"])
    parser.add_argument("--limit", type=int, default=888)
    parser.add_argument("--out-dir", default="artifacts/phase3/full-validation")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.limit, args.threads, args.batch_size, args.bootstrap_reps) < 1 or len(set(args.models)) != len(args.models):
        parser.error("Counts must be positive and models must be unique")
    if args.worker:
        if len(args.models) != 1:
            parser.error("Worker requires exactly one model")
        run_model(args, args.models[0])
    else:
        for model in args.models:
            command = [sys.executable, "-u", "-m", "citewise.research_benchmark", "--worker", "--models", model,
                       "--limit", str(args.limit), "--out-dir", args.out_dir, "--data-dir", args.data_dir,
                       "--threads", str(args.threads), "--batch-size", str(args.batch_size), "--device", args.device]
            subprocess.run(command, check=True)
        assemble(args)


if __name__ == "__main__":
    main()
