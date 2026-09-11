# Lifelong Memory Architecture

## Core problem

The system may accumulate millions or billions of experiences over years. Those experiences cannot all be placed into active context at once.

The solution is a searchable external memory environment rather than proportional growth of the base model.

## Memory classes

### Working memory

Current task state, active hypotheses, recent observations, and immediately relevant information.

### Episodic memory

Records of what happened during previous tasks.

### Semantic memory

Generalized concepts extracted from repeated experiences.

### Procedural memory

Reusable methods and successful strategies.

### World-state memory

Time-sensitive facts with validity periods, evidence, and confidence.

## Memory record example

```yaml
id: 482921
content: "Fedora NVIDIA driver failed after kernel update."
type: episodic
entities: [Fedora, NVIDIA, kernel]
outcome: successful
procedure: rebuild-driver-module
confidence: 0.96
created_at: 2026-03-12
status: current
```

## Key principle

Raw experiences remain available as historical evidence, while frequently reused knowledge can be compressed into higher-level concepts.

## Memory is not just storage

The memory system should decide:

- what is worth storing,
- what can be merged,
- what should be summarized,
- what should be archived,
- what should be marked uncertain,
- what should be superseded rather than deleted.
