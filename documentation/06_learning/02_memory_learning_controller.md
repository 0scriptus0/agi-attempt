# Memory Learning Controller

## Responsibility

The memory controller decides when experience should be added, updated, generalized, merged, or rejected.

## Candidate operations

```text
ADD
UPDATE
MERGE
SUPERSEDE
ARCHIVE
REJECT
```

## Example decision process

```text
new experience
  -> worth remembering?
  -> duplicate?
  -> contradiction?
  -> user-specific or global?
  -> verified?
  -> generalizable?
```

## Global memory boundary

The reasoning model should not have unrestricted authority to rewrite global knowledge.

Global updates should pass through a memory controller with evidence, validation, provenance, and confidence tracking.

## Learning from outcomes

The memory system should record not only information but also how information performed when used.

Example:

```text
procedure: Method A for Error X
applications: 842
successes: 790
failures: 52
success_rate: 93.8%
```

This allows the system to learn conditions under which a procedure succeeds or fails.
