# v4 dataset reproducibility

## Current status

The committed `datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl` artifact contains 315 records. The current `scripts/generate_lewi_v4_quality.py` computes 345 records from its generation loops before reaching `assert len(rows) == 315`.

Therefore the checked-in 315-record artifact is valid as a pinned training artifact, but the current generator is not a byte-for-byte reproducible source for it.

## Required resolution

Before calling v4 reproducible, reconcile the generator and artifact. Prefer regenerating the artifact from the corrected deterministic generator, then validate:

1. generator completes without assertion failure;
2. generated row count is exactly 315;
3. every JSONL record passes the dataset validator;
4. generated artifact matches the committed artifact byte-for-byte, or the artifact is intentionally replaced and the change is documented;
5. CI validates the final contract without generating or mutating repository files.
