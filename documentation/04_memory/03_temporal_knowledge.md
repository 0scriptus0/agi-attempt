# Temporal and Non-Destructive Knowledge Updates

## Principle

Memory should not be destructively overwritten when reality changes.

Old knowledge may have been correct at an earlier time or under a previous software version.

## Superseding model

```text
OLD FACT
status: superseded
valid_until: version 4
        |
        v
NEW FACT
status: current
valid_from: version 5
```

## Required metadata

A factual memory should be able to carry:

- creation time,
- last verification time,
- source,
- version/context,
- confidence,
- validity interval,
- status.

## Why this matters

A contradiction does not always mean that one memory is false. It may indicate a change in version, environment, context, or time.

## External reference rule

The system should never automatically rewrite long-term truth merely because an external search produced a conflicting result.

Instead it should create a conflict, investigate it, and potentially record a newer time-scoped fact.
