"""Precompute a pinned model's paragraph vectors for a single official split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .benchmark import memory_snapshot, system_metadata
from .dataset import load_papers
from .retrieval import MODEL_IDS, MODEL_REVISIONS, embedded_paper, get_model


def prepare(model_name, data_dir, split, cache_dir, output):
    data_path = Path(data_dir) / f"{split}.jsonl"
    cache_dir = Path(cache_dir)
    os.environ["CITEWISE_EMBED_CACHE_DIR"] = str(cache_dir)
    os.environ.pop("CITEWISE_DISABLE_DISK_EMBED_CACHE", None)
    papers = list(load_papers(data_path))
    get_model(model_name)
    rows = []
    started = time.perf_counter()
    for paper in papers:
        if not paper["paragraphs"]:
            continue
        paper_start = time.perf_counter()
        embedded_paper(model_name, paper["id"], tuple(p["text"] for p in paper["paragraphs"]))
        rows.append({"paper_id": paper["id"], "paragraph_count": len(paper["paragraphs"]),
                     "elapsed_seconds": time.perf_counter() - paper_start})
        embedded_paper.cache_clear()
    report = {"split": split, "model": model_name, "model_id": MODEL_IDS[model_name],
              "model_revision": MODEL_REVISIONS[model_name], "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "paper_count": len(papers), "embedded_paper_count": len(rows),
              "paragraph_count": sum(r["paragraph_count"] for r in rows),
              "elapsed_seconds": time.perf_counter() - started, "peak_memory": memory_snapshot(),
              "hardware_and_software": system_metadata(), "papers": rows}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REVISIONS))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="data/cache/embeddings")
    parser.add_argument("--output", default="artifacts/embedding-precompute.json")
    args = parser.parse_args()
    print(json.dumps(prepare(args.model, args.data_dir, args.split, args.cache_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
