# Hierarchical Retrieval

## Motivation

Flat similarity retrieval can become inefficient and noisy as memory grows.

A hierarchical system narrows the search space progressively.

## Example hierarchy

```text
all experience
    |
    +-- domain
          |
          +-- topic
                |
                +-- episode cluster
                      |
                      +-- individual episode
```

## Retrieval process

```text
broad query
  -> relevant domain
  -> relevant topic
  -> relevant memory cluster
  -> relevant episodes
```

## Benefits

- fewer candidates require expensive reranking,
- less irrelevant context is passed to the reasoner,
- search can use different indexes at different levels,
- the memory corpus can grow without linearly increasing active context.

## Research target

Benchmark hierarchical retrieval against flat retrieval on:

- recall of relevant memories,
- retrieval latency,
- tokens returned,
- downstream task accuracy,
- robustness to contradictory memories.
