"""Prepare QASPER without inventing examples or replacing official paper splits."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
QASPER_REVISION = "91bc63e221475d59a6e17e4299fbfc5a919d7c7a"


def sequence_records(obj):
    """HF `Sequence` fields can be lists of structs OR structs of aligned lists."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        sizes = {len(v) for v in obj.values() if isinstance(v, list)}
        if not sizes:
            return []
        if len(sizes) > 1:
            raise ValueError(f"Unexpected inconsistent aligned columns: {sizes}")
        n = next(iter(sizes))
        return [{k: v[i] if isinstance(v, list) else v for k, v in obj.items()} for i in range(n)]
    raise TypeError(f"Expected list/dict, got {type(obj).__name__}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def process_paper(row: dict, split: str):
    pid = str(row["id"])
    paragraphs = []
    by_normalized = {}
    sections = sequence_records(row.get("full_text"))
    for sec_idx, section in enumerate(sections):
        for para_idx, raw in enumerate(section.get("paragraphs") or []):
            txt = clean_text(raw)
            if not txt:
                continue
            paragraph_id = f"{pid}:s{sec_idx}:p{para_idx}"
            paragraphs.append({"id": paragraph_id, "section": section.get("section_name", ""), "text": txt})
            by_normalized.setdefault(normalize(txt), set()).add(paragraph_id)

    questions = []
    for qa in sequence_records(row.get("qas")):
        evidence = set()
        raw_evidence_count = 0
        unsupported_float = 0
        unanswerable_votes = 0
        answers = sequence_records(qa.get("answers"))
        total_votes = 0
        for annotation in answers:
            # QASPER HF can expose each per-question annotation as a nested
            # `answer` list, even though each response is a logical answer.
            raw_answer = annotation.get("answer") or {}
            actual_answers = raw_answer if isinstance(raw_answer, list) else ([raw_answer] if isinstance(raw_answer, dict) else [])
            for answer in actual_answers:
                total_votes += 1
                if answer.get("unanswerable") is True:
                    unanswerable_votes += 1
                    continue
                for phrase in answer.get("evidence") or []:
                    raw_evidence_count += 1
                    if str(phrase).startswith("FLOAT SELECTED"):
                        unsupported_float += 1
                        continue
                    evidence.update(by_normalized.get(normalize(phrase), []))
        if total_votes and unanswerable_votes == total_votes:
            status = "unanswerable"
        elif evidence:
            status = "scorable"
        else:
            status = "unmatched_or_no_text_evidence"
        questions.append({
            "id": str(qa.get("question_id")), "text": clean_text(qa.get("question", "")),
            "gold_paragraph_ids": sorted(evidence), "status": status,
            "annotated_evidence_items": raw_evidence_count,
            "unsupported_figure_items": unsupported_float,
        })
    return {"id": pid, "split": split, "title": clean_text(row.get("title", "")),
            "abstract": clean_text(row.get("abstract", "")), "paragraphs": paragraphs,
            "questions": questions}


def load_papers(path: str | Path):
    """Each JSONL record is one paper; paper boundaries remain intact."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(destination: str | Path, limit_per_split: int | None = None):
    from datasets import load_dataset

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    all_papers = set()
    summary = {"source": "allenai/qasper", "revision": QASPER_REVISION, "license": "CC BY 4.0",
               "method": "Official paper-level splits, no resplitting", "splits": {}}
    for split in SPLITS:
        counter = Counter()
        path = destination / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for i, original in enumerate(load_dataset("allenai/qasper", "qasper", split=split, revision=QASPER_REVISION)):
                if limit_per_split is not None and i >= limit_per_split:
                    break
                paper = process_paper(original, split)
                if paper["id"] in all_papers:
                    raise ValueError(f"Leakage: paper {paper['id']} occurs in multiple splits")
                all_papers.add(paper["id"])
                counter["papers"] += 1
                counter["paragraphs"] += len(paper["paragraphs"])
                counter["questions"] += len(paper["questions"])
                for q in paper["questions"]:
                    counter[f"questions_{q['status']}"] += 1
                    counter["unsupported_figure_items"] += q["unsupported_figure_items"]
                out.write(json.dumps(paper, ensure_ascii=False) + "\n")
        summary["splits"][split] = dict(counter)
    (destination / "data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
