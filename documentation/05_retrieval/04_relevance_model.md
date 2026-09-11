# Relevance Model

## Problem

A memory can be topically similar without being appropriate for the exact task.

## Three useful relevance concepts

### Broad relevance

The memory concerns the same general topic.

### Situational relevance

The memory concerns a closely matching environment, conditions, or problem.

### Exact precedent

The memory concerns essentially the same conditions and has a verified prior outcome.

## Example

```text
broad_match: 0.91
situational_match: 0.63
exact_match: 0.12
verified_success: 0.97
```

The reasoner should use these as separate signals rather than collapsing everything into one similarity score.

## Confidence dimensions

A useful memory record should distinguish:

- truth confidence,
- applicability confidence,
- evidence strength,
- freshness,
- verification status.

A memory can be highly reliable in general but poorly applicable to a specific task.
