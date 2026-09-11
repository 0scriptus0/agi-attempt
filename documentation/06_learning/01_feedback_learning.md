# Feedback and Continual Learning

## Goal

Turn task outcomes and corrections into useful experience without blindly rewriting global knowledge.

## Post-task signals

A task can produce structured outcomes:

```text
succeeded: true / false / unknown
factually_correct: true / false / unknown
user_satisfied: true / false / unknown
externally_verified: true / false / unknown
```

The `unknown` state is essential because the system should not force certainty when the evidence is insufficient.

## Correction pipeline

```text
output
  -> user feedback or external outcome
  -> candidate correction
  -> verification
  -> scope classification
  -> memory update
```

## Example

A user says:

> "That is incorrect. The actual issue is Z."

The system should create a candidate correction rather than immediately changing global knowledge.

It can then:

1. search documentation,
2. reproduce the issue,
3. compare independent evidence,
4. evaluate the correction,
5. update memory only if justified.

## Learning rule

A user correction is evidence. It is not automatically truth.
