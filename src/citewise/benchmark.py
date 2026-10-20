"""Offline, phase-separated retrieval benchmark on validation papers only."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import load_papers
from .evaluation import metrics_for_ranking
from .retrieval import MODEL_IDS, embedded_paper, get_model, get_supervised_ranker, rank


def memory_snapshot():
    """Return current and process-peak RSS in bytes when the OS exposes it."""
    if os.name == "nt":
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if ok:
            return {"rss_bytes": int(counters.WorkingSetSize),
                    "peak_rss_bytes": int(counters.PeakWorkingSetSize), "source": "Windows GetProcessMemoryInfo"}
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != "Darwin":
            peak *= 1024
        return {"rss_bytes": None, "peak_rss_bytes": int(peak), "source": "resource.getrusage"}
    except (ImportError, AttributeError, OSError):
        return {"rss_bytes": None, "peak_rss_bytes": None, "source": None}


def total_memory_bytes():
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = Status()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def system_metadata(probe_cuda=True):
    metadata = {"os": platform.platform(), "processor": platform.processor(),
                "cpu_count": os.cpu_count(), "python": platform.python_version(),
                "total_physical_memory_bytes": total_memory_bytes(), "packages": {}}
    for package in ("numpy", "scikit-learn", "sentence-transformers", "torch", "transformers",
                    "huggingface-hub", "datasets", "fastapi"):
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][package] = None
    if not probe_cuda:
        metadata.update({"cuda_available": None, "gpu": None, "cuda_probe_performed": False})
        return metadata
    try:
        import torch
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        metadata["cuda_available"] = False
        metadata["gpu"] = None
    return metadata


def paired_bootstrap(query_rows, model_a, model_b, metric, reps=2000, seed=42):
    """Paired mean difference with paper-cluster resampling; difference is A minus B."""
    a = {(r["paper_id"], r["question_id"]): r for r in query_rows if r["model"] == model_a}
    b = {(r["paper_id"], r["question_id"]): r for r in query_rows if r["model"] == model_b}
    if not a or a.keys() != b.keys():
        raise ValueError("Paired bootstrap requires identical non-empty query sets")
    grouped = defaultdict(list)
    for key, row in a.items():
        grouped[key[0]].append(float(row[metric]) - float(b[key][metric]))
    papers = sorted(grouped)
    point = statistics.mean(v for group in grouped.values() for v in group)
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(papers, size=len(papers), replace=True)
        values = [value for paper_id in draw for value in grouped[paper_id]]
        samples[i] = float(np.mean(values))
    low, high = np.quantile(samples, [.025, .975])
    return {"model_a": model_a, "model_b": model_b, "metric": metric,
            "mean_difference_a_minus_b": point, "ci95_low": float(low), "ci95_high": float(high),
            "bootstrap_reps": reps, "paper_clusters": len(papers), "query_count": len(a)}


def _questions(papers, limit):
    all_count = sum(len(p["questions"]) for p in papers)
    scorable = [(p, q) for p in papers for q in p["questions"] if q["status"] == "scorable"]
    if limit > len(scorable):
        raise ValueError(f"Requested {limit} scorable questions, only {len(scorable)} available")
    return all_count, scorable, scorable[:limit]


def _load_model(name):
    if name in MODEL_IDS:
        return get_model(name)
    if name == "supervised-logistic":
        return get_supervised_ranker()
    return None


def run_benchmark(models, limit=40, data_dir="data/processed", out_dir="artifacts/phase2/benchmark",
                  bootstrap_reps=2000):
    papers = list(load_papers(Path(data_dir) / "validation.jsonl"))
    total_questions, all_scorable, queries = _questions(papers, limit)
    paper_map = {p["id"]: p for p in papers}
    unique_eval_papers = {p["id"] for p, _ in queries}
    runtime_start = time.perf_counter()
    if any(name in MODEL_IDS for name in models):
        import sentence_transformers  # Load shared runtime code outside per-checkpoint timings.
    if "supervised-logistic" in models:
        import joblib
        import sklearn
    runtime_initialization_ms = (time.perf_counter() - runtime_start) * 1000
    rows, failures, summaries, phases = [], [], [], {}
    for model_name in models:
        if model_name == "bm25":
            load_ms = 0.0
            load_mem_start = load_mem_end = memory_snapshot()
        else:
            get_model.cache_clear()
            embedded_paper.cache_clear()
            if model_name == "supervised-logistic":
                get_supervised_ranker.cache_clear()
            load_mem_start = memory_snapshot()
            start = time.perf_counter()
            _load_model(model_name)
            load_ms = (time.perf_counter() - start) * 1000
            load_mem_end = memory_snapshot()

        paragraph_embedding_ms = 0.0
        embedding_phase = []
        query_groups = defaultdict(list)
        for paper, question in queries:
            query_groups[paper["id"]].append((paper, question))
        model_rows, latencies = [], []
        for paper_id, group in query_groups.items():
            paper = paper_map[paper_id]
            if model_name in MODEL_IDS and paper["paragraphs"]:
                started = time.perf_counter()
                embedded_paper(model_name, paper_id,
                               tuple(p["text"] for p in paper["paragraphs"]))
                elapsed = (time.perf_counter() - started) * 1000
                paragraph_embedding_ms += elapsed
                embedding_phase.append({"paper_id": paper_id, "paragraph_count": len(paper["paragraphs"]),
                                        "embedding_ms": elapsed})
            for _, question in group:
                retrieved, elapsed = rank(question["text"], paper, model_name, top_k=5)
                score = metrics_for_ranking([r["id"] for r in retrieved],
                                            question["gold_paragraph_ids"], k=5)
                row = {"model": model_name, "paper_id": paper_id, "question_id": question["id"],
                       "question": question["text"], "latency_ms": round(elapsed, 4),
                       "recall_at_5": score["recall"], "mrr_at_5": score["mrr"],
                       "ndcg_at_5": score["ndcg"],
                       "gold_ids": "|".join(question["gold_paragraph_ids"]),
                       "retrieved_ids": "|".join(r["id"] for r in retrieved)}
                model_rows.append(row)
                latencies.append(elapsed)
                if score["recall"] == 0:
                    failures.append(row)
        rows.extend(model_rows)
        sorted_latencies = sorted(latencies)
        p95 = sorted_latencies[min(len(sorted_latencies) - 1, math.ceil(.95 * len(sorted_latencies)) - 1)]
        summaries.append({"model": model_name, "questions_evaluated": len(model_rows), "k": 5,
                          "recall_at_k": statistics.mean(r["recall_at_5"] for r in model_rows),
                          "mrr_at_k": statistics.mean(r["mrr_at_5"] for r in model_rows),
                          "ndcg_at_k": statistics.mean(r["ndcg_at_5"] for r in model_rows),
                          "p50_latency_ms": statistics.median(latencies), "p95_latency_ms": p95,
                          "latency_basis": "query encoding and ranking; model loaded and paper embeddings cached"})
        phases[model_name] = {"model_loading_ms": load_ms,
                              "paragraph_embedding_ms_total": paragraph_embedding_ms,
                              "paragraphs_embedded": sum(x["paragraph_count"] for x in embedding_phase),
                              "papers_embedded": len(embedding_phase),
                              "embedding_by_paper": embedding_phase,
                              "memory_after_model_load": load_mem_end,
                              "memory_after_embedding_and_queries": memory_snapshot()}

    pairs = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            for metric in ("recall_at_5", "mrr_at_5", "ndcg_at_5"):
                pairs.append(paired_bootstrap(rows, model_a, model_b, metric, reps=bootstrap_reps))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence = {"dataset": "allenai/qasper", "split": "validation",
                "selection_policy": "train fit; validation for decisions; test not read",
                "limit": limit, "paper_count": len(papers), "question_count": total_questions,
                "scorable_question_count": len(all_scorable), "evaluated_question_count": len(queries),
                "evaluated_paper_count": len(unique_eval_papers), "k": 5,
                "hardware_and_software": system_metadata(),
                "runtime_initialization_ms": runtime_initialization_ms,
                "memory_peak_scope": "whole benchmark process; peak is cumulative across selected models",
                "process_memory_peak": memory_snapshot(), "summaries": summaries,
                "phases": phases, "paired_comparisons": pairs,
                "note": "HF_HUB_OFFLINE=1. Shared runtime imports are timed separately; model loading uses local cache and excludes downloads/retries."}
    (out / "benchmark_validation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    for name, data in (("per_query_validation.csv", rows), ("failure_cases_validation.csv", failures),
                       ("paired_comparisons_validation.csv", pairs)):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]) if data else [])
            if data:
                writer.writeheader()
                writer.writerows(data)
            else:
                writer.writeheader()
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Offline phase-separated benchmark on validation only")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--models", nargs="+", choices=["bm25", "supervised-logistic", *MODEL_IDS],
                        default=["bm25", "minilm", "bge-base"])
    parser.add_argument("--limit", type=int, default=40, help="Bounded count of scorable validation questions")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--out-dir", default="artifacts/phase2/benchmark")
    args = parser.parse_args()
    if args.limit <= 0 or args.bootstrap_reps <= 0:
        parser.error("--limit and --bootstrap-reps must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"  # Cache-only run: downloads and retries are not timing evidence.
    print(json.dumps(run_benchmark(args.models, args.limit, args.data_dir, args.out_dir,
                                   args.bootstrap_reps), indent=2))


if __name__ == "__main__":
    main()
