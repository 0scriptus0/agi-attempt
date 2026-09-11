# Memory Deduplication and Compression

## Problem

A lifelong agent could encounter the same error or fact hundreds or thousands of times. Storing every occurrence as equally important creates unnecessary retrieval noise.

## Multi-stage deduplication

```text
new memory
  -> exact hash check
  -> semantic similarity check
  -> entity / metadata comparison
  -> contradiction check
  -> merge, link, or preserve as distinct
```

## Memory cluster model

Instead of storing 1,000 identical failures as 1,000 equally active records:

```text
Memory Cluster
  concept: Python dependency incompatibility
  occurrences: 1243
  supporting cases: 1180
  contradicting cases: 63
  successful procedures: ...
  confidence: ...
```

The raw episodes remain archived, but active retrieval primarily uses the compressed representation.

## Important distinction

Exact text deduplication is necessary but insufficient. Semantically equivalent memories can use different wording.

Therefore the system should combine lexical, embedding, entity, metadata, and outcome-based signals.

## Compression objective

Compression should preserve:

- useful evidence,
- exceptions,
- causal clues,
- successful and failed procedures,
- temporal boundaries,
- uncertainty.

It must not reduce every repeated experience to an overgeneralized rule.
