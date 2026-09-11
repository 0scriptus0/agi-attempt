# Memory Search Engine

## Core idea

A billion-memory system does not need a billion-memory context. It needs an efficient search system.

The conceptual analogy is a search engine:

```text
huge information corpus
  -> indexing
  -> candidate retrieval
  -> ranking
  -> evidence selection
  -> reasoning
```

## Multiple retrieval modes

The memory system should support more than generic semantic similarity.

Possible search modes:

```text
memory.search_semantic()
memory.search_exact()
memory.search_entity()
memory.search_time()
memory.search_procedure()
memory.search_outcome()
memory.search_causal()
```

## Search as an active reasoning process

For difficult tasks, the model should be able to search, inspect results, identify missing information, and search again.

```text
query
  -> retrieve
  -> inspect
  -> identify gap
  -> reformulate
  -> retrieve again
  -> compare
```

This is closer to research than one-shot RAG.
