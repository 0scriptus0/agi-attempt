#!/usr/bin/env python3

"""
Lewi 1 Dataset Audit

A read-only quantitative audit across the six dataset files, run AFTER
dataset_validate.py passes structurally. This does not modify anything.

Reports, per dataset and across datasets:
- record counts
- exact duplicate records
- duplicate ids
- duplicate "input"-equivalent fields
- duplicate "answer"-equivalent fields (assistant_response / content)
- cross-dataset id collisions
- cross-dataset input/content overlap
- distribution of domain / decision / memory_action / memory_type
- concentration of the most common answer text (to catch the
  "generic templated answer repeated across many prompts" pattern)

Usage:
    python tools/dataset_audit.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "self": ROOT / "datasets" / "domains" / "self_v2_aligned.jsonl",
    "reasoning": ROOT / "datasets" / "reasoning" / "reasoning_agentic_sft.jsonl",
    "combined_sft": ROOT / "datasets" / "domains" / "combined_sft.jsonl",
    "memory": ROOT / "datasets" / "domains" / "memory_annotation_sft.jsonl",
    "preference": ROOT / "datasets" / "domains" / "preference_pairs.jsonl",
    "heldout_eval": ROOT / "datasets" / "domains" / "heldout_eval.jsonl",
}


def load(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def answer_field(record: dict[str, Any]) -> str | None:
    """Pick whichever field represents this record's 'answer' text."""
    for key in ("assistant_response", "content", "chosen", "prompt"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return None


def prompt_field(record: dict[str, Any]) -> str | None:
    """Pick whichever field represents this record's 'prompt/input' text."""
    for key in ("input", "content", "prompt"):
        if key in record and isinstance(record[key], str):
            return record[key]
    return None


def main() -> None:
    all_records: dict[str, list[dict[str, Any]]] = {}
    for name, path in DATASETS.items():
        if path.exists():
            all_records[name] = load(path)

    print("=" * 70)
    print("LEWI 1 DATASET AUDIT")
    print("=" * 70)

    # ---- Per-dataset stats ----
    global_id_locations: dict[str, list[str]] = defaultdict(list)
    global_input_locations: dict[str, list[str]] = defaultdict(list)

    for name, records in all_records.items():
        print()
        print("-" * 70)
        print(f"{name}  ({len(records)} records)")
        print("-" * 70)

        # Exact duplicate whole records (by canonical JSON, ignoring key order)
        exact_seen: dict[str, int] = {}
        exact_dupes = 0
        for r in records:
            key = json.dumps(r, sort_keys=True)
            if key in exact_seen:
                exact_dupes += 1
            exact_seen[key] = exact_seen.get(key, 0) + 1
        print(f"Exact duplicate records: {exact_dupes}")

        # Duplicate ids (within this file)
        id_counts = Counter(r.get("id") for r in records if "id" in r)
        dup_ids = {k: v for k, v in id_counts.items() if v > 1}
        print(f"Duplicate ids within file: {len(dup_ids)}")
        if dup_ids:
            for k, v in list(dup_ids.items())[:5]:
                print(f"    id={k!r} appears {v} times")

        # Duplicate prompt/input text (within this file)
        prompt_counts = Counter()
        for r in records:
            p = prompt_field(r)
            if p:
                prompt_counts[p] += 1
        dup_prompts = {k: v for k, v in prompt_counts.items() if v > 1}
        print(f"Duplicate prompt/input text within file: {len(dup_prompts)} distinct values "
              f"({sum(dup_prompts.values())} records involved)")

        # Duplicate answer text (within this file)
        answer_counts = Counter()
        for r in records:
            a = answer_field(r)
            if a:
                answer_counts[a] += 1
        dup_answers = {k: v for k, v in answer_counts.items() if v > 1}
        total_dup_answer_records = sum(dup_answers.values())
        print(f"Duplicate answer text within file: {len(dup_answers)} distinct values "
              f"({total_dup_answer_records} records involved, "
              f"{100 * total_dup_answer_records / max(len(records),1):.0f}% of file)")

        if dup_answers:
            most_common_answer, most_common_count = answer_counts.most_common(1)[0]
            print(f"Most repeated answer text ({most_common_count}x): "
                  f"{most_common_answer[:100]!r}{'...' if len(most_common_answer) > 100 else ''}")

        # Distributions (only meaningful for sft-shaped / memory-shaped records)
        for field in ("domain", "decision", "memory_action", "memory_type", "scope"):
            values = [r[field] for r in records if field in r and r[field]]
            if values:
                dist = Counter(values)
                print(f"{field} distribution ({len(dist)} distinct): "
                      f"{dict(dist.most_common(8))}"
                      + (" ..." if len(dist) > 8 else ""))

        # Feed cross-dataset trackers
        for r in records:
            rid = r.get("id")
            if rid:
                global_id_locations[rid].append(name)
            p = prompt_field(r)
            if p:
                global_input_locations[p].append(name)

    # ---- Cross-dataset overlap ----
    print()
    print("=" * 70)
    print("CROSS-DATASET OVERLAP")
    print("=" * 70)

    cross_id = {k: v for k, v in global_id_locations.items() if len(set(v)) > 1}
    print(f"IDs appearing in more than one dataset: {len(cross_id)}")
    for k, v in list(cross_id.items())[:10]:
        print(f"    id={k!r} in {v}")

    cross_input = {k: v for k, v in global_input_locations.items() if len(set(v)) > 1}
    print(f"Identical prompt/input text appearing in more than one dataset: {len(cross_input)}")
    for k, v in list(cross_input.items())[:10]:
        print(f"    {k[:80]!r}... in {v}")

    print()
    print("=" * 70)
    print("AUDIT COMPLETE (read-only — nothing was modified)")
    print("=" * 70)


if __name__ == "__main__":
    main()
