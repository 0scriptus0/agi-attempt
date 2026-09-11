# Agentic Reasoning Loop

## Core loop

```text
REASON
  -> PLAN
  -> ACT
  -> OBSERVE
  -> VERIFY
  -> UPDATE STATE
  -> REPLAN OR CONTINUE
```

## Why agentic reasoning differs from static reasoning

A static model produces an answer from the information already available to it.

An agent can change its information state and environment by acting. Therefore reasoning and action must be interleaved.

## State components

The agent should explicitly track:

- goal,
- constraints,
- world state,
- plan state,
- evidence state,
- tool results,
- unresolved questions,
- failures,
- confidence.

## Failure recovery

A failed action should not automatically trigger an identical retry.

Instead:

```text
failure
  -> diagnose
  -> identify what changed or was learned
  -> choose a new action
  -> continue or re-plan
```

## Termination

The agent should stop when:

- success criteria are satisfied,
- remaining uncertainty is immaterial,
- expected information gain is lower than cost,
- or the task is genuinely unsolvable with available information.

Correct abstention is a valid outcome.
