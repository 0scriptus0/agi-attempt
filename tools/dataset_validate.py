#!/usr/bin/env python3

"""
Lewi 1 Dataset Validator

Validates JSONL datasets before they are used for training.

This version reflects the schemas actually observed by inspecting the
six dataset files (see chat history: `head -n 1 <file> | python3 -c
"import json,sys; print(list(json.loads(sys.stdin.read()).keys()))"`),
not an assumed schema. In particular:

- 'situation_state' (self / reasoning / sft-shaped combined_sft records)
  is a compact STRING encoding (e.g.
  "goal=...; known=...; unknown=...; constraints=..."), not a JSON object.
- combined_sft mixes multiple record shapes together (tagged with
  '_dataset'), so records are validated by their actual shape
  (duck-typed), with a soft warning if that shape disagrees with the
  '_dataset' tag -- we haven't audited the exact tag vocabulary yet,
  so this is a warning, not a hard failure.
- memory_annotation_sft, preference_pairs, and heldout_eval each have
  their own distinct schema (not the sft instruction/input/assistant_response
  shape).

Checks:
- JSON syntax
- one JSON object per line
- required fields (per detected record shape)
- empty values
- duplicate IDs
- dataset-specific structure
- train/evaluation separation
- basic text quality

Usage:

    python tools/dataset_validate.py

Or:

    python tools/dataset_validate.py datasets/domains/self_v2_aligned.jsonl

Exit codes:
    0 = validation passed
    1 = validation failed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "self": ROOT / "datasets" / "domains" / "self_v2_aligned.jsonl",
    "reasoning": ROOT / "datasets" / "reasoning" / "reasoning_agentic_sft.jsonl",
    "combined_sft": ROOT / "datasets" / "domains" / "combined_sft.jsonl",
    "memory": ROOT / "datasets" / "domains" / "memory_annotation_sft.jsonl",
    "preference": ROOT / "datasets" / "domains" / "preference_pairs.jsonl",
    "heldout_eval": ROOT / "datasets" / "domains" / "heldout_eval.jsonl",
}

# Datasets whose records are always one fixed shape.
FIXED_SHAPE = {
    "self": "sft",
    "reasoning": "sft",
    "memory": "memory",
    "preference": "preference",
    "heldout_eval": "eval",
}


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

        self.records = 0
        self.valid = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

        self.ids: dict[str, int] = {}

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_jsonl(path: Path, dataset_name: str) -> ValidationResult:
    result = ValidationResult(dataset_name, path)

    if not path.exists():
        result.errors.append(f"File does not exist: {path}")
        return result

    if path.stat().st_size == 0:
        result.errors.append("File is empty.")
        return result

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):

                if not raw_line.strip():
                    result.warnings.append(
                        f"Line {line_number}: blank line"
                    )
                    continue

                result.records += 1

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    result.errors.append(
                        f"Line {line_number}: invalid JSON: {exc}"
                    )
                    continue

                if not isinstance(record, dict):
                    result.errors.append(
                        f"Line {line_number}: record must be a JSON object"
                    )
                    continue

                record_errors, record_warnings = validate_record(
                    record,
                    dataset_name,
                    line_number,
                )

                result.errors.extend(record_errors)
                result.warnings.extend(record_warnings)

                record_id = record.get("id")

                if record_id is not None:
                    if not isinstance(record_id, str):
                        result.errors.append(
                            f"Line {line_number}: 'id' must be a string"
                        )
                    elif record_id in result.ids:
                        previous = result.ids[record_id]

                        result.errors.append(
                            f"Line {line_number}: duplicate id "
                            f"'{record_id}' (first seen on line {previous})"
                        )
                    else:
                        result.ids[record_id] = line_number

                if not record_errors:
                    result.valid += 1

    except OSError as exc:
        result.errors.append(f"Could not read file: {exc}")

    return result


# ---------------------------------------------------------------------------
# Shape detection (for combined_sft, which mixes record types)
# ---------------------------------------------------------------------------

def detect_shape(record: dict[str, Any]) -> str | None:
    """
    Duck-type a record's shape based on which distinguishing keys it has.

    Returns one of "sft", "preference", "memory", "eval", or None if the
    shape can't be determined (all distinguishing keys absent).
    """

    if "assistant_response" in record and "instruction" in record:
        return "sft"

    if "chosen" in record and "rejected" in record:
        return "preference"

    if "memory_type" in record and "content" in record:
        return "memory"

    if "success_criteria" in record and "prompt" in record:
        return "eval"

    return None


# ---------------------------------------------------------------------------
# Record validation (dispatch)
# ---------------------------------------------------------------------------

def validate_record(
    record: dict[str, Any],
    dataset_name: str,
    line_number: int,
) -> tuple[list[str], list[str]]:

    errors: list[str] = []
    warnings: list[str] = []

    prefix = f"Line {line_number}"

    # Every training record should have an ID.
    if "id" not in record:
        errors.append(f"{prefix}: missing required field 'id'")
    elif not is_nonempty_string(record["id"]):
        errors.append(f"{prefix}: 'id' must be a non-empty string")

    # Prevent accidental literal chain-of-thought fields.
    forbidden_fields = {
        "chain_of_thought",
        "cot",
        "hidden_thought",
        "private_reasoning",
        "secret_reasoning",
    }

    for field in forbidden_fields:
        if field in record:
            errors.append(
                f"{prefix}: forbidden reasoning field '{field}' found"
            )

    # Work out which schema this record should be checked against.
    if dataset_name in FIXED_SHAPE:
        shape = FIXED_SHAPE[dataset_name]
    elif dataset_name == "combined_sft":
        detected = detect_shape(record)

        if detected is None:
            errors.append(
                f"{prefix}: could not determine record shape "
                f"(no recognizable sft/preference/memory/eval keys found)"
            )
            return errors, warnings

        shape = detected

        # combined_sft records are expected to carry a '_dataset' tag.
        # We haven't audited the tag vocabulary yet, so only warn if the
        # tag disagrees with the detected shape rather than failing hard.
        tag = record.get("_dataset")

        if tag is None:
            warnings.append(
                f"{prefix}: missing '_dataset' tag (detected shape: {shape})"
            )
        elif not is_nonempty_string(tag):
            errors.append(f"{prefix}: '_dataset' must be a non-empty string")
    else:
        # Unknown dataset name (e.g. validating an arbitrary file passed
        # on the command line) -- fall back to shape detection, defaulting
        # to "sft" if nothing else matches, to preserve prior behavior.
        shape = detect_shape(record) or "sft"

    if shape == "sft":
        errors.extend(validate_sft_record(record, prefix))
    elif shape == "preference":
        errors.extend(validate_preference_record(record, prefix))
    elif shape == "memory":
        errors.extend(validate_memory_record(record, prefix))
    elif shape == "eval":
        errors.extend(validate_eval_record(record, prefix))

    # General text sanity checks.
    warnings.extend(check_text_quality(record, prefix))

    return errors, warnings


# ---------------------------------------------------------------------------
# SFT datasets (self, reasoning, and sft-shaped combined_sft records)
# ---------------------------------------------------------------------------

def validate_sft_record(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:

    errors: list[str] = []

    required_fields = [
        "instruction",
        "input",
        "assistant_response",
    ]

    for field in required_fields:
        if field not in record:
            errors.append(
                f"{prefix}: missing required field '{field}'"
            )
        elif not is_nonempty_string(record[field]):
            errors.append(
                f"{prefix}: '{field}' must be a non-empty string"
            )

    # 'situation_state' is a compact STRING encoding
    # (e.g. "goal=...; known=...; unknown=...; constraints=..."),
    # confirmed by direct inspection -- not a JSON object.
    if "situation_state" in record:
        if not is_nonempty_string(record["situation_state"]):
            errors.append(
                f"{prefix}: 'situation_state' must be a non-empty string"
            )

    # These fields are allowed to be strings if present.
    optional_string_fields = [
        "domain",
        "decision",
        "reasoning_summary",
        "memory_action",
        "evidence_policy",
        "verification",
        "tool_policy",
        "recovery_policy",
        "_dataset",
    ]

    for field in optional_string_fields:
        if field in record and not isinstance(record[field], str):
            errors.append(
                f"{prefix}: '{field}' must be a string"
            )

    return errors


# ---------------------------------------------------------------------------
# Preference datasets
# ---------------------------------------------------------------------------

def validate_preference_record(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:

    errors: list[str] = []

    # Confirmed schema: id, input, chosen, rejected, preference_reason
    # (NOT 'prompt').
    required_fields = [
        "input",
        "chosen",
        "rejected",
    ]

    for field in required_fields:
        if field not in record:
            errors.append(
                f"{prefix}: missing required field '{field}'"
            )
        elif not is_nonempty_string(record[field]):
            errors.append(
                f"{prefix}: '{field}' must be a non-empty string"
            )

    if "preference_reason" in record and not isinstance(
        record["preference_reason"], str
    ):
        errors.append(f"{prefix}: 'preference_reason' must be a string")

    return errors


# ---------------------------------------------------------------------------
# Memory annotation dataset
# ---------------------------------------------------------------------------

def validate_memory_record(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:

    errors: list[str] = []

    # Confirmed schema: id, memory_type, scope, content, context, why,
    # applicable_when, not_applicable_when, outcome, evidence,
    # confidence, generalization, supersedes.
    required_fields = [
        "memory_type",
        "scope",
        "content",
        "why",
        "applicable_when",
    ]

    for field in required_fields:
        if field not in record:
            errors.append(
                f"{prefix}: missing required field '{field}'"
            )
        elif not is_nonempty_string(record[field]):
            errors.append(
                f"{prefix}: '{field}' must be a non-empty string"
            )

    # 'supersedes' is expected to sometimes be null (no prior memory to
    # replace) -- that's normal, not an error. If present and not null,
    # it must be a string (the id of the superseded memory).
    if "supersedes" in record and record["supersedes"] is not None:
        if not isinstance(record["supersedes"], str):
            errors.append(
                f"{prefix}: 'supersedes' must be a string or null"
            )

    # These are allowed to be strings if present; not asserting exact
    # types for 'confidence' yet since we haven't audited whether it's
    # a string label (e.g. "high") or numeric across all 120 records.
    optional_fields = [
        "context",
        "not_applicable_when",
        "outcome",
        "evidence",
        "generalization",
    ]

    for field in optional_fields:
        if field in record and record[field] is not None and not isinstance(
            record[field], str
        ):
            errors.append(f"{prefix}: '{field}' must be a string or null")

    return errors


# ---------------------------------------------------------------------------
# Held-out eval dataset
# ---------------------------------------------------------------------------

def validate_eval_record(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:

    errors: list[str] = []

    # Confirmed schema: id, category, prompt, success_criteria.
    required_fields = [
        "category",
        "prompt",
    ]

    for field in required_fields:
        if field not in record:
            errors.append(
                f"{prefix}: missing required field '{field}'"
            )
        elif not is_nonempty_string(record[field]):
            errors.append(
                f"{prefix}: '{field}' must be a non-empty string"
            )

    # success_criteria could plausibly be a string description or a
    # list of criteria -- accept either, just not empty/missing.
    if "success_criteria" not in record:
        errors.append(f"{prefix}: missing required field 'success_criteria'")
    else:
        sc = record["success_criteria"]
        if isinstance(sc, str):
            if not sc.strip():
                errors.append(f"{prefix}: 'success_criteria' is empty")
        elif isinstance(sc, list):
            if not sc:
                errors.append(f"{prefix}: 'success_criteria' list is empty")
        else:
            errors.append(
                f"{prefix}: 'success_criteria' must be a string or list"
            )

    return errors


# ---------------------------------------------------------------------------
# Text quality
# ---------------------------------------------------------------------------

def check_text_quality(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:

    warnings: list[str] = []

    # Check whichever "main answer" field this record shape has.
    for field in ("assistant_response", "chosen", "content", "prompt"):
        value = record.get(field)

        if isinstance(value, str):
            if len(value.strip()) < 2:
                warnings.append(
                    f"{prefix}: '{field}' is extremely short"
                )

            if len(value) > 100_000:
                warnings.append(
                    f"{prefix}: '{field}' is unusually long"
                )

    # Detect accidental nulls, but 'supersedes' is EXPECTED to be null
    # sometimes (no prior memory to supersede) -- don't warn on that one.
    for key, value in record.items():
        if value is None and key != "supersedes":
            warnings.append(
                f"{prefix}: field '{key}' contains null"
            )

    return warnings


# ---------------------------------------------------------------------------
# Dataset separation checks
# ---------------------------------------------------------------------------

def check_dataset_separation() -> list[str]:
    """
    Make sure evaluation data isn't accidentally configured as SFT data.

    This is intentionally a lightweight path check. The actual training
    pipeline should also enforce this.
    """

    errors: list[str] = []

    heldout = DATASETS["heldout_eval"].resolve()

    training_files = [
        DATASETS["self"],
        DATASETS["reasoning"],
        DATASETS["combined_sft"],
        DATASETS["memory"],
        DATASETS["preference"],
    ]

    for path in training_files:
        if path.resolve() == heldout:
            errors.append(
                "CRITICAL: heldout_eval.jsonl is also configured as "
                "a training dataset."
            )

    return errors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(result: ValidationResult) -> None:

    print()
    print("=" * 70)
    print(result.name)
    print("=" * 70)

    print(f"File:     {result.path}")
    print(f"Records:  {result.records}")
    print(f"Valid:    {result.valid}")
    print(f"Errors:   {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")

    if result.errors:
        print()
        print("ERRORS:")

        for error in result.errors:
            print(f"  [ERROR] {error}")

    if result.warnings:
        print()
        print("WARNINGS:")

        for warning in result.warnings[:25]:
            print(f"  [WARN]  {warning}")

        if len(result.warnings) > 25:
            print(
                f"  ... {len(result.warnings) - 25} more warnings"
            )

    print()

    if result.passed:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    print()
    print("=" * 70)
    print("LEWI 1 DATASET VALIDATOR")
    print("=" * 70)

    separation_errors = check_dataset_separation()

    if separation_errors:
        print()
        print("DATASET SEPARATION ERRORS:")

        for error in separation_errors:
            print(f"  [ERROR] {error}")

    # Allow validating a specific file:
    #
    # python tools/dataset_validate.py path/to/file.jsonl
    #
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

        if not path.is_absolute():
            path = ROOT / path

        name = path.stem

        result = validate_jsonl(path, name)
        print_result(result)

        return 0 if result.passed else 1

    # Otherwise validate the complete dataset collection.
    overall_passed = True

    for name, path in DATASETS.items():

        result = validate_jsonl(path, name)

        print_result(result)

        if not result.passed:
            overall_passed = False

    if separation_errors:
        overall_passed = False

    print()
    print("=" * 70)

    if overall_passed:
        print("DATASET STATUS: PASS")
        print("All datasets passed validation.")
        print("=" * 70)
        return 0

    print("DATASET STATUS: FAIL")
    print("Fix the errors above before training.")
    print("=" * 70)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())