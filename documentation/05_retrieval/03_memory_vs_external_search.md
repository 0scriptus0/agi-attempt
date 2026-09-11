# Memory Search vs External Search

## Principle

The system should use internal experience and external information for different purposes.

### Memory search

Answers:

> What have I experienced before?

It is fast, cheap, and personalized, but can contain stale, incomplete, or incorrect information.

### External search / documentation

Answers:

> What does the current external evidence say?

It is potentially fresher and independently sourced, but source quality varies and the information may still be wrong or incomplete.

## Recommended architecture

```text
TASK
  -> memory search
  -> external search when warranted
  -> compare evidence
  -> verify
  -> reason
```

## Important rule

External search should not automatically overwrite memory.

Instead, disagreements create evidence conflicts that can trigger verification and time-scoped updates.

## When to search externally

External verification is especially valuable when information is:

- time-sensitive,
- version-dependent,
- high-risk,
- disputed,
- unfamiliar,
- or beyond the agent's verified experience.
