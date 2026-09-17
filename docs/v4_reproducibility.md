# v4 dataset reproducibility

## Current status

The committed `datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl` artifact contains 315 records. The corrected `scripts/generate_lewi_v4_quality.py` deterministically generates the same 315-record artifact.

The generator supports `LEWI_V4_OUTPUT` so reproducibility can be checked against a temporary file without mutating the checked-in artifact. `scripts/check_v4_generator_reproducibility.py` now performs the byte-for-byte comparison and SHA-256 check.

## Acceptance checks

1. generator completes without assertion failure;
2. generated row count is exactly 315;
3. every JSONL record passes the dataset validator;
4. regenerated output matches the committed artifact byte-for-byte;
5. CI validates the final contract without committing or mutating repository files.
