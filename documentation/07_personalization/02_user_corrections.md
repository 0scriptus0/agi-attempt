# User Corrections Without Global Contamination

## Principle

A user can correct the system about their own context without changing general knowledge.

## Example

A user says:

> "I use environment X, not environment Y."

This should update user-scoped context.

A user says:

> "Your explanation of protocol Z is factually wrong."

This should create a global knowledge candidate, not an immediate global rewrite.

## Classification flow

```text
user feedback
  -> identify subject
  -> determine scope
  -> candidate update
  -> verify if global claim
  -> commit to appropriate scope
```

## Principle

Personalization changes how the agent behaves for a user. It should not silently change what the agent believes is true about the world.
