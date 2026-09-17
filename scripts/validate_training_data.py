#!/usr/bin/env python3
"""Validate Lewi training data and configuration without model dependencies."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "train.yaml"
ALLOWED_ACTIONS = {"reason", "memory_search", "memory_write", "code_exec", "file_read", "file_write", "final"}
V4_GENERATED_SFT = "datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl"
V4_GENERATED_PREF = "datasets/reasoning/lewi_agentic_state_transition_preference_v4_generated.jsonl"
V4_GENERATED_CATEGORIES = {"tool_selection", "recovery", "replanning", "memory_read_write", "verification", "termination", "state_propagation", "uncertainty"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: value = json.loads(line)
        except json.JSONDecodeError as exc: raise AssertionError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict): raise AssertionError(f"{path}:{line_no}: record must be an object")
        records.append(value)
    return records


def yaml_paths() -> tuple[list[str], list[str], list[str]]:
    section = None; sft: list[str] = []; preference: list[str] = []; evaluation: list[str] = []
    for raw in CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "sft:": section = sft
        elif line == "preference:": section = preference
        elif line == "evaluation:": section = evaluation
        elif line.startswith("- ") and section is not None: section.append(line[2:].strip())
    return sft, preference, evaluation


def parse_action(text: str) -> dict | None:
    try: action = json.loads(text)
    except json.JSONDecodeError: return None
    if not isinstance(action, dict) or action.get("action") not in ALLOWED_ACTIONS: return None
    return action


def validate_sft(path: Path) -> int:
    records = load_jsonl(path)
    for record in records:
        if "trajectory" in record:
            trajectory = record["trajectory"]
            assert isinstance(trajectory, list) and trajectory, f"{path}: trajectory must be non-empty"
            assert all(isinstance(turn, dict) for turn in trajectory), f"{path}: invalid trajectory turn"
            roles = [turn.get("role") for turn in trajectory]
            assert roles[0] == "user", f"{path}: trajectory must start with user"
            assert all(role in {"user", "assistant"} for role in roles), f"{path}: invalid trajectory role"
            assert all(str(turn.get("content", "")).strip() for turn in trajectory), f"{path}: empty trajectory content"
            for turn in trajectory:
                if turn["role"] == "assistant":
                    assert parse_action(str(turn.get("content", ""))) is not None, f"{path}: invalid trajectory action"
        elif {"instruction", "input", "assistant_response"}.issubset(record):
            assert str(record["assistant_response"]).strip(), f"{path}: assistant response is empty"
        elif {"input", "assistant_response"}.issubset(record):
            assert str(record["input"]).strip() and str(record["assistant_response"]).strip(), f"{path}: empty input/response"
        elif {"memory_type", "content"}.issubset(record):
            assert str(record.get("content", "")).strip(), f"{path}: memory content is empty"
        else:
            raise AssertionError(f"{path}: unsupported SFT schema: {record.keys()}")
    return len(records)


def validate_preferences(path: Path) -> int:
    records = load_jsonl(path)
    for record in records:
        assert {"chosen", "rejected"}.issubset(record), f"{path}: incomplete preference record"
        if "input" not in record: assert "prompt" in record, f"{path}: preference needs input or prompt"
        prompt = str(record.get("input", record.get("prompt", ""))).strip()
        chosen, rejected = str(record["chosen"]).strip(), str(record["rejected"]).strip()
        assert prompt and chosen and rejected and chosen != rejected, f"{path}: invalid preference fields"
        chosen_action, rejected_action = parse_action(chosen), parse_action(rejected)
        if chosen_action is not None or rejected_action is not None:
            assert chosen_action is not None and rejected_action is not None, f"{path}: mixed action/free-form pair"
            assert chosen_action != rejected_action, f"{path}: equivalent JSON preference pair"
    return len(records)


def validate_generated_v4(root: Path, counts: dict[str, int]) -> None:
    sft_path = root / V4_GENERATED_SFT; pref_path = root / V4_GENERATED_PREF
    assert V4_GENERATED_SFT in counts and V4_GENERATED_PREF in counts, "generated v4 datasets must be configured"
    assert counts[V4_GENERATED_SFT] == 315, f"generated v4 SFT must contain 315 records, got {counts[V4_GENERATED_SFT]}"
    assert counts[V4_GENERATED_PREF] == 100, f"generated v4 preference must contain 100 records, got {counts[V4_GENERATED_PREF]}"
    sft_records = load_jsonl(sft_path)
    ids = [str(r.get("id", "")) for r in sft_records]
    assert all(ids) and len(ids) == len(set(ids)), "generated v4 SFT ids must be unique"
    categories = {str(r.get("category", "")) for r in sft_records}
    assert V4_GENERATED_CATEGORIES <= categories, f"generated v4 SFT missing categories: {sorted(V4_GENERATED_CATEGORIES - categories)}"
    pref_records = load_jsonl(pref_path)
    pref_ids = [str(r.get("id", "")) for r in pref_records]
    assert all(pref_ids) and len(pref_ids) == len(set(pref_ids)), "generated v4 preference ids must be unique"


def main() -> None:
    assert CONFIG.exists(), f"missing {CONFIG}"
    sft, preference, evaluation = yaml_paths()
    assert sft and preference and evaluation, "train.yaml sections are incomplete"
    counts = {}
    for relative in sft:
        path = ROOT / relative; assert path.exists(), f"missing SFT source: {relative}"; counts[relative] = validate_sft(path)
    for relative in preference:
        path = ROOT / relative; assert path.exists(), f"missing preference source: {relative}"; counts[relative] = validate_preferences(path)
    for relative in evaluation:
        path = ROOT / relative; assert path.exists(), f"missing evaluation source: {relative}"; load_jsonl(path)
    v4 = ROOT / "datasets/reasoning/lewi_agentic_state_transition_sft_v4.jsonl"
    assert counts[str(v4.relative_to(ROOT))] >= 30, "v4 SFT dataset unexpectedly shrank"
    v4_pref = ROOT / "datasets/reasoning/lewi_agentic_state_transition_preference_v4.jsonl"
    assert counts[str(v4_pref.relative_to(ROOT))] >= 30, "v4 preference dataset unexpectedly shrank"
    validate_generated_v4(ROOT, counts)
    print("Training-data validation passed")
    for path, count in counts.items(): print(f"  {count:4d}  {path}")


if __name__ == "__main__": main()
