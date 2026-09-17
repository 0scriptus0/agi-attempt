#!/usr/bin/env python3
"""Guardrail for the generated v4 SFT dataset.

The committed artifact is intentionally treated as a pinned training input.
This check makes the generator/artifact relationship explicit without silently
rewriting the dataset during CI.
"""
from pathlib import Path

ARTIFACT = Path("datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl")
GENERATOR = Path("scripts/generate_lewi_v4_quality.py")
EXPECTED_ROWS = 315


def main() -> None:
    rows = ARTIFACT.read_text(encoding="utf-8").splitlines()
    assert len(rows) == EXPECTED_ROWS, (len(rows), EXPECTED_ROWS)
    source = GENERATOR.read_text(encoding="utf-8")
    # The current generator contains an assertion for the intended artifact
    # size. This guard prevents accidental changes to either contract from
    # going unnoticed; exact byte-for-byte regeneration is a separate task.
    assert "assert len(rows) == 315" in source
    print(f"v4 generated artifact contract OK: {len(rows)} rows")


if __name__ == "__main__":
    main()
