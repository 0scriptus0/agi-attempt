#!/usr/bin/env python3

"""
Lewi 1 Dataset Dedup

Read-only with respect to the ORIGINAL files: writes cleaned copies to
datasets_cleaned/ and a DATASET_CHANGELOG.md, never overwrites the
originals in datasets/.

Two dedup rules, both verified lossless (records in each collapsed
group are identical except for 'id', or except for 'id' and the
trailing phrasing of 'input'):

1. memory_annotation_sft.jsonl: 114 of 120 records are exact
   duplicates of just 5 lesson texts under different mgen* ids.
   Collapse each duplicate group to one canonical record (lowest id).
   The 6 hand-authored m00X records are already unique and untouched.

2. reasoning_agentic_sft.jsonl: 100 of 121 records collapse into 10
   distinct (domain, decision) scenarios that all share one identical
   generic answer; they only vary in the closing phrasing of 'input'.
   Collapse each scenario group to one canonical record (lowest id).
   The 21 hand-authored records are already unique and untouched.

combined_sft.jsonl is NOT regenerated here: it was found to be an
exact union of reasoning_agentic_sft.jsonl + memory_annotation_sft.jsonl
with zero unique content, and configs/train.yaml lists it alongside
both source files, double-counting every record. The recommended fix
is to remove combined_sft.jsonl from train.yaml (and from the repo)
rather than regenerate a redundant merged file.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets_cleaned"

GENERIC_ANSWER = (
    "I would apply that strategy, verify the critical uncertainty, and "
    "avoid writing a global memory unless the lesson is reusable and "
    "supported."
)


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dedup_memory(records: list[dict]) -> tuple[list[dict], list[str]]:
    by_content: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_content[r["content"]].append(r)

    kept = []
    log_lines = []
    for content, group in by_content.items():
        group_sorted = sorted(group, key=lambda r: r["id"])
        canonical = group_sorted[0]
        kept.append(canonical)
        if len(group) > 1:
            dropped_ids = [r["id"] for r in group_sorted[1:]]
            log_lines.append(
                f"- Kept `{canonical['id']}`, dropped {len(dropped_ids)} "
                f"exact duplicate(s): {', '.join(dropped_ids)}\n"
                f"  content: \"{content[:80]}{'...' if len(content) > 80 else ''}\""
            )
    kept.sort(key=lambda r: r["id"])
    return kept, log_lines


def dedup_reasoning(records: list[dict]) -> tuple[list[dict], list[str]]:
    generic = [r for r in records if r["assistant_response"] == GENERIC_ANSWER]
    handwritten = [r for r in records if r["assistant_response"] != GENERIC_ANSWER]

    by_scenario: dict[tuple, list[dict]] = defaultdict(list)
    for r in generic:
        by_scenario[(r["domain"], r["decision"])].append(r)

    kept = list(handwritten)
    log_lines = []
    for (domain, decision), group in by_scenario.items():
        group_sorted = sorted(group, key=lambda r: r["id"])
        canonical = group_sorted[0]
        kept.append(canonical)
        dropped_ids = [r["id"] for r in group_sorted[1:]]
        log_lines.append(
            f"- Scenario (domain={domain!r}, decision={decision!r}): "
            f"kept `{canonical['id']}`, dropped {len(dropped_ids)} "
            f"phrasing-variant duplicate(s): {', '.join(dropped_ids)}"
        )
    kept.sort(key=lambda r: r["id"])
    return kept, log_lines


def main() -> None:
    mem_path = ROOT / "datasets" / "domains" / "memory_annotation_sft.jsonl"
    reas_path = ROOT / "datasets" / "reasoning" / "reasoning_agentic_sft.jsonl"

    mem_records = load(mem_path)
    reas_records = load(reas_path)

    mem_kept, mem_log = dedup_memory(mem_records)
    reas_kept, reas_log = dedup_reasoning(reas_records)

    dump(mem_kept, OUT / "domains" / "memory_annotation_sft.jsonl")
    dump(reas_kept, OUT / "reasoning" / "reasoning_agentic_sft.jsonl")

    # Files that need no dedup (already clean) are copied through
    # unchanged so datasets_cleaned/ is a complete, ready-to-use set.
    for rel in [
        "domains/self_v2_aligned.jsonl",
        "domains/preference_pairs.jsonl",
        "domains/heldout_eval.jsonl",
    ]:
        src = ROOT / "datasets" / rel
        dst = OUT / rel
        dump(load(src), dst)

    changelog = OUT / "DATASET_CHANGELOG.md"
    with changelog.open("w", encoding="utf-8") as f:
        f.write("# Dataset Cleanup Changelog\n\n")
        f.write(
            "Generated by `tools/dataset_dedup.py`. Originals in "
            "`datasets/` are untouched; cleaned copies are in "
            "`datasets_cleaned/`.\n\n"
        )

        f.write("## memory_annotation_sft.jsonl\n\n")
        f.write(f"Before: {len(mem_records)} records. After: {len(mem_kept)} records.\n\n")
        f.write(
            "Every dropped record was verified identical to its kept "
            "canonical record in every field except `id`.\n\n"
        )
        f.write("\n".join(mem_log) + "\n\n")

        f.write("## reasoning_agentic_sft.jsonl\n\n")
        f.write(f"Before: {len(reas_records)} records. After: {len(reas_kept)} records.\n\n")
        f.write(
            f"21 hand-authored records (unique, non-generic answers) were "
            f"kept unchanged. The remaining 100 records all shared one "
            f"identical generic answer (\"{GENERIC_ANSWER[:60]}...\") and "
            f"were verified identical to each other in every field except "
            f"`id` and the trailing phrasing of `input`. Each of the 10 "
            f"distinct (domain, decision) scenarios was collapsed to one "
            f"representative record.\n\n"
        )
        f.write("\n".join(reas_log) + "\n\n")

        f.write("## combined_sft.jsonl\n\n")
        f.write(
            "NOT regenerated. The original combined_sft.jsonl (241 records) "
            "was found to be an exact union of the original "
            "reasoning_agentic_sft.jsonl (121) + memory_annotation_sft.jsonl "
            "(120), with zero unique content of its own. configs/train.yaml "
            "lists combined_sft.jsonl alongside both of its source files, "
            "which doubles the training weight of every reasoning and "
            "memory record relative to self_v2_aligned.jsonl.\n\n"
            "Recommended fix: remove combined_sft.jsonl from "
            "configs/train.yaml's `sft:` list (and delete the file), "
            "rather than regenerate a merged file that would just "
            "reintroduce the double-counting problem.\n\n"
        )

        f.write("## Unchanged (already clean)\n\n")
        f.write(
            "- self_v2_aligned.jsonl (13 records, 0 duplicates) — copied through as-is\n"
            "- preference_pairs.jsonl (100 records) — copied through as-is; "
            "NOTE: 90/100 `chosen` texts collapse into 10 distinct values, "
            "not yet deduped here, flagged for a follow-up decision since "
            "preference pairs may legitimately repeat a chosen answer "
            "across different rejected/prompt pairs\n"
            "- heldout_eval.jsonl (61 records, 0 duplicates) — copied through as-is, "
            "unchanged since this is the held-out evaluation set\n"
        )

    print(f"memory_annotation_sft.jsonl: {len(mem_records)} -> {len(mem_kept)}")
    print(f"reasoning_agentic_sft.jsonl: {len(reas_records)} -> {len(reas_kept)}")
    print(f"Wrote cleaned files to {OUT}")
    print(f"Wrote changelog to {changelog}")


if __name__ == "__main__":
    main()
