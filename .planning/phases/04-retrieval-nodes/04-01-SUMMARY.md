---
phase: 04-retrieval-nodes
plan: "01"
subsystem: retrieval
tags: [query-planning, vector-search, fastembed, type-filtering]

dependency-graph:
  requires:
    - 03-01 (graph store Protocol and PPR)
    - 03-02 (FAISS vector store)
  provides:
    - Query complexity classification (SIMPLE/MODERATE/COMPLEX)
    - Entity extraction from natural language queries
    - Text-to-embedding vector search
    - Post-retrieval type filtering
  affects:
    - 04-02 (graph retrieval will consume RankedComponent)
    - 04-03 (score fusion will merge vector + graph results)

tech-stack:
  added: []
  patterns:
    - Module-level singleton for expensive model initialization
    - isinstance narrowing for Protocol implementation access
    - Post-filtering with over-fetch (3x) for type constraints

key-files:
  created:
    - src/skill_retriever/nodes/retrieval/__init__.py
    - src/skill_retriever/nodes/retrieval/models.py
    - src/skill_retriever/nodes/retrieval/query_planner.py
    - src/skill_retriever/nodes/retrieval/vector_search.py
    - tests/test_query_planner.py
    - tests/test_vector_search.py
  modified: []

decisions:
  - key: entity-extraction-via-isinstance
    choice: Use isinstance(graph_store, NetworkXGraphStore) for _graph access
    reason: Protocol doesn't expose iteration; isinstance enables type narrowing for internal access
    alternatives: [add iterator method to Protocol, accept NetworkXGraphStore directly]

  - key: type-filter-post-retrieval
    choice: Filter by component type AFTER vector retrieval with 3x over-fetch
    reason: Research showed filtering after score fusion preserves semantic relevance better
    alternatives: [pre-filter via separate indexes, filter in vector store]

  - key: lazy-embedding-init
    choice: Module-level _embedding_model with _get_embedding_model() accessor
    reason: TextEmbedding model is expensive to create (~2-3s); lazy init avoids import-time cost
    alternatives: [dependency injection, per-call initialization]

metrics:
  duration: 8m
  completed: 2026-02-03
---

# Phase 04 Plan 01: Query Planner + Vector Search Node Summary

Query classification and vector search with lazy embedding initialization and post-retrieval type filtering.

## Objective

Create the query planner and vector search retrieval node for Phase 4. Enable natural language search over the component graph with query complexity classification to optimize retrieval strategy downstream.

## What Was Built

### Task 1: Retrieval Models and Query Planner

**models.py** - Core data structures:
- `QueryComplexity(StrEnum)`: SIMPLE, MODERATE, COMPLEX classification
- `RetrievalPlan` dataclass: Holds complexity, PPR settings, flow pruning flag, max results
- `RankedComponent` Pydantic model: component_id, score, rank, source (vector/graph/fused)

**query_planner.py** - Heuristic-based query analysis:
- `STOPWORDS` frozenset: Common English words to filter from entity extraction
- `plan_retrieval(query, entity_count)`: Classifies queries based on length and entity count
  - SIMPLE: < 300 chars AND <= 2 entities (skip PPR, alpha=0.85, max=10)
  - COMPLEX: > 600 chars OR > 5 entities (PPR + flow pruning, alpha=0.7, max=30)
  - MODERATE: Everything else (PPR only, alpha=0.85, max=20)
- `extract_query_entities(query, graph_store)`: Tokenize, filter stopwords, match against graph node labels (case-insensitive)

### Task 2: Vector Search Node

**vector_search.py** - Text-to-embedding search:
- `_embedding_model` module-level singleton with lazy initialization
- `_get_embedding_model()`: Returns cached TextEmbedding instance
- `search_by_text(query, vector_store, top_k)`: Generate embedding, search FAISS, return RankedComponent list
- `search_with_type_filter(query, vector_store, graph_store, component_type, top_k)`: Fetch 3x results, filter by type, re-rank

## Tests Added

**test_query_planner.py** (11 tests):
1. test_short_query_few_entities - SIMPLE classification
2. test_medium_query_several_entities - MODERATE classification
3. test_long_query_many_entities - COMPLEX classification
4. test_long_query_alone_triggers_complex - Length alone triggers COMPLEX
5. test_many_entities_alone_triggers_complex - Entity count alone triggers COMPLEX
6. test_stopwords_filtered - Stopword filtering works
7. test_case_insensitive_matching - Case-insensitive label matching
8. test_uppercase_query - Uppercase query tokens match lowercase labels
9. test_no_matches - No matches returns empty set
10. test_only_stopwords - Only stopwords returns empty set
11. test_empty_graph - Empty graph returns empty set

**test_vector_search.py** (7 tests):
1. test_returns_ranked_component - Returns RankedComponent with source="vector"
2. test_sorted_descending - Scores sorted descending, ranks sequential
3. test_filters_by_type - Type filter returns only requested type
4. test_reranks_after_filter - Re-ranking works after filtering
5. test_none_type_returns_all - None type returns all types
6. test_empty_vector_store - Empty store returns empty list
7. test_empty_with_type_filter - Empty stores with filter returns empty list

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

1. **Entity extraction via isinstance**: Used isinstance(graph_store, NetworkXGraphStore) to access _graph for node iteration, since GraphStore Protocol doesn't expose iteration methods.

2. **Type filter post-retrieval with 3x over-fetch**: Fetches 3x top_k candidates before filtering by type, ensuring enough results survive filtering while preserving semantic relevance ordering.

3. **Lazy embedding model initialization**: TextEmbedding model created on first use via module-level singleton pattern, avoiding 2-3s import-time cost.

## Technical Notes

- pyright ignore for `reportPrivateUsage` when accessing `_graph` through isinstance-narrowed type
- pyright ignore for fastembed's missing type stubs (reportMissingTypeStubs)
- pyright ignore for unknown types from fastembed embed() method

## Verification Results

```
uv run pytest tests/test_query_planner.py tests/test_vector_search.py -v
# 18 passed in 13.15s

uv run pyright src/skill_retriever/nodes/retrieval/
# 0 errors, 0 warnings, 0 informations

uv run ruff check src/skill_retriever/nodes/retrieval/
# All checks passed!

uv run python -c "from skill_retriever.nodes.retrieval import QueryComplexity, plan_retrieval, search_by_text; print('Imports OK')"
# Imports OK
```

## Next Phase Readiness

Ready for 04-02 (Graph Retrieval Node):
- RankedComponent model ready for graph results
- RetrievalPlan provides PPR settings for graph traversal
- extract_query_entities provides seed nodes for PPR

Dependencies resolved:
- GraphStore Protocol from 03-01
- FAISSVectorStore from 03-02
- ComponentType from 02-01
