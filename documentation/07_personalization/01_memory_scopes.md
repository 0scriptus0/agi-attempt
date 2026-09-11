# Memory Scopes and Personalization

## Problem

A lifelong agent must not confuse one user's preference or correction with universal truth.

## Four memory scopes

```text
TASK
SESSION
USER
GLOBAL
```

### Task scope

Only relevant to the current task.

### Session scope

Useful during the current conversation or active project.

### User scope

Long-term preferences, habits, and user-specific facts.

### Global scope

Knowledge or procedures intended to generalize beyond one user.

## Example

User statement:

> "I prefer concise responses."

Correct representation:

```text
scope: USER
type: preference
content: response_length = concise
```

Incorrect representation:

```text
GLOBAL FACT:
concise responses are preferred
```

## Isolation principle

A memory should never silently change scope.

Moving information from USER to GLOBAL requires a separate validation process.
