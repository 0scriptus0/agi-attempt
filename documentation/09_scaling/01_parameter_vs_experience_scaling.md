# Parameter Scaling vs Experience Scaling

## Problem

Continually adding learned information directly to the model risks making the model increasingly expensive to train and serve.

## Desired scaling law

We want lifetime experience to scale much faster than active model parameters.

Example target:

```text
Model parameters:     approximately fixed
Lifetime experiences: 1,000,000 -> 500,000,000+
Active context:       bounded
Per-task compute:     bounded or slowly increasing
```

The exact numbers are hypothetical. The architectural target is the important part.

## Design strategy

Use:

- external long-term memory,
- searchable indexes,
- hierarchical retrieval,
- deduplication,
- semantic compression,
- offline consolidation,
- adaptive compute.

## Key principle

Do not confuse storage growth with neural parameter growth.

A large searchable information environment can coexist with a relatively stable reasoning model.
