# Adaptive Compute and Reasoning Budget

## Motivation

Using maximum reasoning compute on every task is inefficient. Using minimum compute on every task causes failures on difficult tasks.

The reasoning controller should allocate computation dynamically.

## Conceptual policy

```text
if easy and high confidence:
    answer
elif moderate uncertainty:
    think longer
elif ambiguity is high:
    branch hypotheses
elif information is missing:
    search or use tools
elif correctness is testable:
    verify
else:
    answer with calibrated uncertainty
```

## Information-value model

A useful conceptual objective is:

```text
expected utility = expected progress + expected information gain - compute cost - risk
```

This is not required to be the exact implemented formula. It is the design principle for the controller.

## Research target

Train or engineer a lightweight controller that predicts which additional reasoning action has the highest expected value for the current task.
