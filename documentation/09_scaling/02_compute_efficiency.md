# Compute Efficiency

## Expensive computation should be selective

The system should reserve large reasoning budgets for tasks where the expected benefit is high.

Cheap operations should handle common cases:

- hashing,
- indexing,
- metadata filtering,
- approximate nearest-neighbor retrieval,
- lightweight classification,
- duplicate detection.

Expensive operations should be reserved for:

- difficult reasoning,
- ambiguous evidence,
- high-value search,
- contradiction resolution,
- verification,
- complex planning.

## Offline vs online work

Not every memory operation needs to happen during a user interaction.

Offline jobs can perform:

- clustering,
- summarization,
- graph construction,
- deduplication,
- concept extraction,
- statistics updates,
- memory consolidation.

This keeps online interactions responsive.

## Research target

Measure:

```text
task success / token
memory retrievals / task
external searches / task
wall-clock latency
total inference compute
```

Efficiency is a first-class metric, not an afterthought.
