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
CHECKPOINT_IO_VERSION = 2
LEGACY_RUNNER_SHA256 = "7d0aca1947835ed810ea958ff86fcaa8714549e58923cbb704d00d100d56c64e"
LEGACY_RETRIEVAL_SHA256 = "f942d37b791fbabc74104b77f566e034cec385f0a688fad09bc15d46c2e1bd39"
LEGACY_EVALUATION_SHA256 = "983591430454575e93f9874c8d41f19693a3ec5c38a43d0997df278879b96a41"
LEGACY_SOURCE_SHA256 = "ec7384454bb461d8452a0aadfbc259f36969540cd368b42d19bed0920f5ececd"
LEGACY_FULL_RUN_CONFIG_ID = "72a0c22ef939960521c2ae3362b854b24c52a20f0a042f8321b146537ba36907"
PHASE8_FULL_RUN_SOURCE_SHA256 = "6619d499876cb3d3e6cf55a3d43d2d5f89ba953eb357926464ad599b04a237c8"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    ready = False
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"sha256": digest(payload), "payload": payload},
                                    ensure_ascii=False, indent=2, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        ready = True
        _replace_checkpoint(temp, path)
    except Exception:
        if not ready:
            temp.unlink(missing_ok=True)
        raise


def _replace_checkpoint(temp, path):
    for attempt in range(6):
        try:
            os.replace(temp, path)
            return
        except PermissionError as exc:
            if (os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33}
                    or attempt == 5):
                raise
            time.sleep(.1 * (2 ** attempt))


def read_checkpoint(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("sha256") != digest(record["payload"]):
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return record["payload"]


def _checkpoint_writer_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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


def _legacy_checkpoint_source_digest():
    root = Path(__file__).parent
    current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("retrieval.py", "evaluation.py")}
    if current != {"retrieval.py": LEGACY_RETRIEVAL_SHA256,
                   "evaluation.py": LEGACY_EVALUATION_SHA256}:
        return None
    legacy = digest({"research_benchmark.py": LEGACY_RUNNER_SHA256, **current})
    return legacy if legacy == LEGACY_SOURCE_SHA256 else None


def _resume_shared_config_matches(previous, requested):
    keys = ("protocol", "data_sha256", "query_plan_sha256", "normalize_embeddings", "precision",
            "similarity", "batch_size", "threads", "device", "blas_threads", "limit", "k")
    if not all(previous.get(key) == requested.get(key) for key in keys):
        return False
    def hardware_without_cuda_probe(config):
        value = dict(config.get("hardware_and_software") or {})
        for key in ("cuda_available", "gpu", "cuda_probe_performed"):
            value.pop(key, None)
        return value
    return hardware_without_cuda_probe(previous) == hardware_without_cuda_probe(requested)


def _same_config_except_source(previous, requested):
    return ({key: value for key, value in previous.items() if key != "source_sha256"}
            == {key: value for key, value in requested.items() if key != "source_sha256"})


def _source_fingerprint_for_resume(config, checkpoints_root):
    """Preserve the source fingerprint for an existing run after checkpoint-only code changes."""
    legacy = _legacy_checkpoint_source_digest()
    allowed = {LEGACY_SOURCE_SHA256, PHASE8_FULL_RUN_SOURCE_SHA256}
    if not legacy or legacy != LEGACY_SOURCE_SHA256:
        return config
    checkpoints_root = Path(checkpoints_root)
    current_model_dir = checkpoints_root / config["model"]
    own_manifests = (current_model_dir / "manifest.json", current_model_dir / "manifest.tmp",
                     *current_model_dir.glob("manifest.json.*.tmp"))
    for path in own_manifests:
        previous = _read_valid_manifest(path)
        if (previous and previous.get("config_id") in {LEGACY_FULL_RUN_CONFIG_ID,
                                                        digest(previous["config"])}
                and previous["config"].get("source_sha256") in allowed
                and _same_config_except_source(previous["config"], config)):
            compatible = dict(config)
            compatible["source_sha256"] = previous["config"]["source_sha256"]
            return compatible

    peers = []
    requested_neural = config.get("model") in MODEL_IDS
    for model_dir in checkpoints_root.glob("*"):
        if model_dir.name == config.get("model"):
            continue
        for path in (model_dir / "manifest.json", model_dir / "manifest.tmp",
                     *model_dir.glob("manifest.json.*.tmp")):
            previous = _read_valid_manifest(path)
            if not previous or previous["config"].get("source_sha256") not in allowed:
                continue
            previous_config = previous["config"]
            if _resume_shared_config_matches(previous_config, config):
                peer_neural = previous_config.get("model") in MODEL_IDS
                peers.append((peer_neural == requested_neural, path.stat().st_mtime_ns,
                              previous_config["source_sha256"]))
    if peers:
        _, _, source_sha = max(peers)
        compatible = dict(config)
        compatible["source_sha256"] = source_sha
        return compatible
    return config


def _paper_checkpoint_temporary_files(path):
    path = Path(path)
    candidates = [path.with_suffix(".tmp"), *path.parent.glob(f"{path.name}.*.tmp")]
    return sorted(set(candidate for candidate in candidates if candidate.is_file()),
                  key=lambda candidate: candidate.stat().st_mtime_ns)


def _manifest_temporary_files(path):
    path = Path(path)
    candidates = [path.with_suffix(".tmp"), *path.parent.glob(f"{path.name}.*.tmp")]
    return sorted(set(candidate for candidate in candidates if candidate.is_file()),
                  key=lambda candidate: candidate.stat().st_mtime_ns)


def _valid_manifest(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        return False
    if payload.get("config_id") != digest(payload["config"]):
        return False
    if payload.get("status") not in {"running", "complete"} or not isinstance(payload.get("attempts"), list):
        return False
    for attempt in payload["attempts"]:
        if (not isinstance(attempt, dict) or not isinstance(attempt.get("id"), str)
                or not isinstance(attempt.get("completed_papers"), int)
                or attempt["completed_papers"] < 0):
            return False
    return True


def _manifest_progress(payload):
    attempts = payload["attempts"]
    if not attempts:
        return (0, 0, 0.0, payload["status"] == "complete")
    last = attempts[-1]
    return (len(attempts), last["completed_papers"], float(last.get("elapsed_seconds", 0.0)),
            payload["status"] == "complete")


def _manifest_is_newer(candidate, current):
    candidate_ids = [attempt["id"] for attempt in candidate["attempts"]]
    current_ids = [attempt["id"] for attempt in current["attempts"]]
    if len(candidate_ids) > len(current_ids) and candidate_ids[:len(current_ids)] == current_ids:
        return True
    return candidate_ids == current_ids and _manifest_progress(candidate) > _manifest_progress(current)


def _read_valid_manifest(path):
    try:
        payload = read_checkpoint(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if _valid_manifest(payload) else None


def _manifest_progress_is_durable(payload, durable_paper_count):
    return all(attempt["completed_papers"] <= durable_paper_count for attempt in payload["attempts"])


def _save_manifest(path, payload):
    saved = dict(payload)
    saved["checkpoint_io_version"] = CHECKPOINT_IO_VERSION
    saved["checkpoint_writer_sha256"] = _checkpoint_writer_sha256()
    save_checkpoint(path, saved)
    return saved


def _recover_manifest(path, config_id, durable_paper_count):
    path = Path(path)
    current = _read_valid_manifest(path) if path.exists() else None
    if current and current["config_id"] != config_id:
        raise ValueError(f"Resume configuration changed for {path.parent.name}; use a NEW output directory")
    candidates = []
    for temp_path in _manifest_temporary_files(path):
        candidate = _read_valid_manifest(temp_path)
        if (candidate and candidate["config_id"] == config_id
                and _manifest_progress_is_durable(candidate, durable_paper_count)):
            candidates.append((temp_path, candidate))
    if current and _manifest_progress_is_durable(current, durable_paper_count):
        chosen_path, chosen = path, current
        for temp_path, candidate in candidates:
            if _manifest_is_newer(candidate, chosen):
                chosen_path, chosen = temp_path, candidate
        if chosen_path == path:
            return current
    else:
        chosen_path, chosen = (None, None)
        if candidates:
            chosen_path, chosen = max(candidates, key=lambda item: _manifest_progress(item[1]))
    if chosen is None:
        if not path.exists() and durable_paper_count == 0:
            return None
        raise ValueError(f"No checksum-valid manifest agrees with {durable_paper_count} durable paper checkpoints: {path}")
    recovered = dict(chosen)
    recovered["recovered_from_manifest"] = chosen_path.name
    recovered["recovered_at"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(path, recovered)
    print(f"Recovered {path.name} from {chosen_path.name}; {durable_paper_count} durable paper checkpoints", flush=True)
    return read_checkpoint(path)


def _validate_paper_checkpoint(payload, model_name, config_id, paper, questions):
    if payload.get("config_id") != config_id or payload.get("paper_id") != paper["id"]:
        raise ValueError("Paper checkpoint configuration or paper ID mismatch")
    if payload.get("paragraph_count") != len(paper["paragraphs"]):
        raise ValueError("Paper checkpoint paragraph count mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or [row.get("question_id") for row in rows] != [question["id"] for question in questions]:
        raise ValueError("Paper checkpoint question sequence mismatch")
    valid_ids = {paragraph["id"] for paragraph in paper["paragraphs"]}
    for row, question in zip(rows, questions):
        if row.get("model") != model_name or row.get("paper_id") != paper["id"]:
            raise ValueError("Paper checkpoint row identity mismatch")
        if row.get("gold_ids") != "|".join(question["gold_paragraph_ids"]):
            raise ValueError("Paper checkpoint gold evidence mismatch")
        retrieved = row.get("retrieved_ids", "").split("|")
        if (not retrieved or len(retrieved) > 5 or len(retrieved) != len(set(retrieved))
                or not set(retrieved) <= valid_ids):
            raise ValueError("Paper checkpoint contains invalid retrieved evidence IDs")
        for metric in ("recall_at_5", "mrr_at_5", "ndcg_at_5"):
            value = float(row[metric])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Paper checkpoint contains invalid {metric}")
    return payload


def _recover_paper_checkpoint(path, model_name, config_id, paper, questions):
    path = Path(path)
    if path.exists():
        return _validate_paper_checkpoint(read_checkpoint(path), model_name, config_id, paper, questions)
    valid = []
    for temp_path in _paper_checkpoint_temporary_files(path):
        try:
            payload = _validate_paper_checkpoint(read_checkpoint(temp_path), model_name, config_id,
                                                 paper, questions)
            valid.append((temp_path, payload))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not valid:
        return None
    if any(payload["rows"] != valid[0][1]["rows"] for _, payload in valid[1:]):
        raise ValueError(f"Conflicting valid temporary paper checkpoints for {paper['id']}")
    temp_path, payload = valid[-1]
    save_checkpoint(path, payload)
    print(f"Recovered paper checkpoint {paper['id']} from {temp_path.name}", flush=True)
    return _validate_paper_checkpoint(read_checkpoint(path), model_name, config_id, paper, questions)


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
    folder = Path(args.out_dir) / "checkpoints" / name
    folder.mkdir(parents=True, exist_ok=True)
    with model_lock(folder / "worker.lock"):
        config = _source_fingerprint_for_resume(config, Path(args.out_dir) / "checkpoints")
        config_id = digest(config)
        manifest_path = folder / "manifest.json"
        existing_manifest = _read_valid_manifest(manifest_path) if manifest_path.exists() else None
        if existing_manifest and existing_manifest["config_id"] != config_id:
            raise ValueError(f"Resume configuration changed for {name}; use a NEW output directory")
        groups = defaultdict(list)
        for paper, question in queries:
            groups[paper["id"]].append((paper, question))
        completed = {}
        for paper_id, group in groups.items():
            path = folder / (hashlib.sha256(paper_id.encode()).hexdigest() + ".json")
            checkpoint = _recover_paper_checkpoint(path, name, config_id, group[0][0],
                                                   [question for _, question in group])
            if checkpoint is not None:
                completed[paper_id] = checkpoint
        manifest = _recover_manifest(manifest_path, config_id, len(completed))
        if manifest is None:
            if completed:
                raise ValueError(f"Paper checkpoints exist without a valid manifest: {manifest_path}")
            manifest = {"config_id": config_id, "config": config, "attempts": [], "status": "running"}
            manifest = _save_manifest(manifest_path, manifest)
        if len(completed) < len(groups):
            attempt = {"id": uuid.uuid4().hex, "started_at": datetime.now(timezone.utc).isoformat(),
                       "runtime_initialization_ms": runtime_ms, "elapsed_seconds": 0.0,
                       "completed_papers": 0, "checkpoint_io_version": CHECKPOINT_IO_VERSION,
                       "checkpoint_writer_sha256": _checkpoint_writer_sha256()}
            manifest["attempts"].append(attempt)
            manifest = _save_manifest(manifest_path, manifest)
            load_start = time.perf_counter()
            model = get_model(name) if name in MODEL_IDS else (get_supervised_ranker() if name == "supervised-logistic" else None)
            attempt["model_loading_ms"] = (time.perf_counter() - load_start) * 1000
            if name in MODEL_IDS:
                attempt["encoder_runtime"] = {"embedding_dimension": model.get_sentence_embedding_dimension(),
                                               "max_seq_length": model.max_seq_length,
                                               "dtype": str(next(model.parameters()).dtype)}
            attempt["memory_after_model_load"] = memory_snapshot()
            attempt["elapsed_seconds"] = time.perf_counter() - started
            manifest = _save_manifest(manifest_path, manifest)
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
                manifest = _save_manifest(manifest_path, manifest)
                print(f"{name}: {len(completed)}/{len(groups)} papers; {sum(len(c['rows']) for c in completed.values())}/{len(queries)} queries", flush=True)
        manifest["status"] = "complete"
        if manifest["attempts"]:
            manifest["attempts"][-1].setdefault("finished_at", datetime.now(timezone.utc).isoformat())
        manifest = _save_manifest(manifest_path, manifest)
        rows = [row for paper_id in groups for row in completed[paper_id]["rows"]]
        result = {"config_id": config_id, "config": config, "attempts": manifest["attempts"], "rows": rows,
                  "papers": list(completed.values()), "total_execution_seconds": sum(a["elapsed_seconds"] for a in manifest["attempts"]),
                  "checkpoint_io_version": CHECKPOINT_IO_VERSION,
                  "checkpoint_writer_sha256": _checkpoint_writer_sha256(),
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


def _assembly_source_compatibility(results):
    signatures = {(r["config"]["data_sha256"], r["config"]["query_plan_sha256"],
                   r["config"]["protocol"]) for r in results}
    if len(signatures) != 1:
        raise ValueError("Refusing to compare different datasets, question plans or protocols")
    sources = {r["config"]["source_sha256"] for r in results}
    source_by_model = {r["config"]["model"]: r["config"]["source_sha256"] for r in results}
    if len(sources) == 1:
        return {"status": "uniform", "source_sha256_by_model": source_by_model}
    if (sources == {LEGACY_SOURCE_SHA256, PHASE8_FULL_RUN_SOURCE_SHA256}
            and _legacy_checkpoint_source_digest() == LEGACY_SOURCE_SHA256):
        mode = "checkpoint_io_migration"
    else:
        raise ValueError(f"Incompatible source fingerprints in results: {sorted(sources)}")

    def comparison_hardware(config):
        hardware = dict(config.get("hardware_and_software") or {})
        for key in ("cuda_available", "gpu", "cuda_probe_performed"):
            hardware.pop(key, None)
        return hardware

    if len({json.dumps(comparison_hardware(r["config"]), sort_keys=True) for r in results}) != 1:
        raise ValueError("Hardware/software environment differs across checkpoint source migration")
    return {"status": mode, "source_sha256_by_model": source_by_model,
            "legacy_source_sha256": LEGACY_SOURCE_SHA256,
            "phase8_run_source_sha256": PHASE8_FULL_RUN_SOURCE_SHA256,
            "checkpoint_writer_sha256": _checkpoint_writer_sha256(),
            "note": "Source hashes differ only across the checkpoint-writer compatibility migration; retrieval/evaluation hashes, data, query plan, protocol and normalized hardware match."}


def assemble(args):
    results = [read_checkpoint(Path(args.out_dir) / "checkpoints" / name / "model.json") for name in args.models]
    source_compatibility = _assembly_source_compatibility(results)
    rows = [row for result in results for row in result["rows"]]
    query_sets = [{(row["paper_id"], row["question_id"]) for row in result["rows"]} for result in results]
    if len({frozenset(query_set) for query_set in query_sets}) != 1:
        raise ValueError("Refusing to compare models with different validation questions")
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
                "source_compatibility": source_compatibility,
                "checkpoint_io_version": CHECKPOINT_IO_VERSION,
                "checkpoint_writer_sha256": _checkpoint_writer_sha256(),
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
    parser.add_argument("--assemble-only", action="store_true",
                        help="Assemble completed per-model checkpoints without running inference")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.limit, args.threads, args.batch_size, args.bootstrap_reps) < 1 or len(set(args.models)) != len(args.models):
        parser.error("Counts must be positive and models must be unique")
    if args.worker:
        if len(args.models) != 1:
            parser.error("Worker requires exactly one model")
        run_model(args, args.models[0])
    elif args.assemble_only:
        assemble(args)
    else:
        for model in args.models:
            command = [sys.executable, "-u", "-m", "citewise.research_benchmark", "--worker", "--models", model,
                       "--limit", str(args.limit), "--out-dir", args.out_dir, "--data-dir", args.data_dir,
                       "--threads", str(args.threads), "--batch-size", str(args.batch_size), "--device", args.device]
            subprocess.run(command, check=True)
        assemble(args)


if __name__ == "__main__":
    main()
