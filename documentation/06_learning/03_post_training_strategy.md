# Post-Training Strategy

## Principle

Post-training should improve the cognitive machinery rather than attempting to permanently encode every individual experience into model parameters.

## What should be learned in weights

- reasoning skills,
- planning policies,
- tool-use policies,
- verification strategies,
- memory usage,
- retrieval decisions,
- uncertainty estimation,
- recovery behavior.

## What should primarily live outside the weights

- individual experiences,
- time-sensitive facts,
- user preferences,
- task histories,
- episodic observations,
- mutable world state.

## Training target

A successful post-training system should make the model better at:

> deciding what to remember, deciding what to retrieve, deciding when to verify, and deciding how to learn from outcomes.

This makes training improve the learning process itself rather than merely adding another pile of knowledge.
