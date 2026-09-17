#!/usr/bin/env python3
"""Guardrail for the generated v4 SFT dataset.

The committed artifact is intentionally treated as a pinned training input.
This check makes the generator/artifact relationship explicit without silently
rewriting the dataset during CI.
"""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile

ARTIFACT = Path("datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl")
GENERATOR = Path("scripts/generate_lewi_v4_quality.py")
EXPECTED_ROWS = 315


def main() -> None:
    rows = ARTIFACT.read_text(encoding="utf-8").splitlines()
    assert len(rows) == EXPECTED_ROWS, (len(rows), EXPECTED_ROWS)
    source = GENERATOR.read_text(encoding="utf-8")
    assert "assert len(rows) == 315" in source
    expected_hash = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / ARTIFACT.name
        env = os.environ.copy()
        env["LEWI_V4_OUTPUT"] = str(generated)
        subprocess.run(["python3", str(GENERATOR)], check=True, env=env)
        assert generated.read_bytes() == ARTIFACT.read_bytes()
        actual_hash = hashlib.sha256(generated.read_bytes()).hexdigest()
    assert actual_hash == expected_hash
    print(f"v4 reproducibility check passed: {len(rows)} rows, sha256={expected_hash}")


if __name__ == "__main__":
    main()
