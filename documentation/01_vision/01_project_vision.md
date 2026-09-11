# Project Vision

## Objective

Build a general-purpose reasoning and agentic system whose capabilities can improve continuously through experience without requiring the core model to grow proportionally with every new piece of information.

## Central hypothesis

A scalable AGI-like architecture should separate:

1. general cognitive capability stored in model parameters,
2. accumulated experience stored in long-term memory,
3. current task state stored in working memory,
4. external reality checked through tools and current sources,
5. learning behavior controlled by a dedicated memory and learning system.

## Why this matters

A conventional approach to continual learning is to repeatedly train a larger or increasingly specialized model. That can make the model more capable, but it risks parameter growth, expensive retraining, catastrophic forgetting, and poor separation between general knowledge and individual experiences.

The alternative is to allow the information environment around the model to grow while keeping the active reasoning core relatively stable.

## Design goal

The lifetime information available to the system should be able to grow by orders of magnitude without requiring the entire lifetime history to fit inside the model context or active inference computation.

## Non-goals

This project should not assume that external memory alone produces AGI. Memory is one component of a broader system involving reasoning, planning, verification, tool use, learning, and generalization.

## Guiding principles

- Evidence over appearance of reasoning.
- Search before stuffing context.
- Verification before high-confidence global updates.
- Preserve history instead of destructively overwriting it.
- Separate user-specific information from global information.
- Spend expensive inference compute only when expected value justifies it.
- Measure progress using reproducible evaluations rather than subjective impressions.
