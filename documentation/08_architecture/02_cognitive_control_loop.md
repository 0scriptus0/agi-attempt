# Cognitive Control Loop

## Closed loop

```text
understand
  -> plan
  -> retrieve
  -> reason
  -> verify
  -> act
  -> observe
  -> update state
  -> learn
  -> repeat
```

## Control decisions

At each cycle the system should be able to decide:

- answer now,
- retrieve memory,
- search externally,
- call a tool,
- branch hypotheses,
- verify,
- backtrack,
- ask for information,
- or terminate.

## Why control matters

A powerful model can still behave inefficiently if it lacks control over when to think, search, act, and stop.

The cognitive controller therefore becomes a first-class architectural component rather than a prompt convention.
