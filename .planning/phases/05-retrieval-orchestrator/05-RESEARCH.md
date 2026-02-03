# Phase 5: Retrieval Orchestrator - Research

**Researched:** 2026-02-03
**Domain:** Pipeline orchestration, transitive dependency resolution, conflict detection, caching, latency optimization
**Confidence:** HIGH

## Summary

Phase 5 coordinates the Phase 4 retrieval nodes (query planner, vector search, PPR engine, flow pruner, score fusion, context assembler) into a unified pipeline that resolves transitive dependencies, detects conflicts, enforces token budgets, and meets strict latency SLAs (500ms simple, 1000ms complex). The codebase already has all building blocks in `src/skill_retriever/nodes/retrieval/`; this phase orchestrates them.

The standard approach uses a coordinator class that chains existing nodes with early-exit optimization (return immediately if simple query hits cache), LRU caching at both query and component levels, and latency monitoring via `time.perf_counter()`. For dependency resolution, NetworkX provides `nx.descendants()` for transitive closure in O(V+E) time. Conflict detection leverages the existing `EdgeType.CONFLICTS_WITH` edges in the graph store.

**Primary recommendation:** Build a `RetrievalPipeline` class in `src/skill_retriever/workflows/` that composes existing nodes synchronously (no async needed at <1000ms target) with `@functools.lru_cache` for query memoization and `nx.descendants()` for dependency resolution.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| NetworkX | 3.x | Transitive closure via `descendants()`, cycle detection | Already in codebase, O(V+E) BFS-based traversal |
| functools.lru_cache | stdlib | Query result caching | Thread-safe, O(1) lookup, built into Python |
| time.perf_counter | stdlib | Latency monitoring | Highest resolution timer, monotonic, platform-independent |
| Pydantic | 2.x | Result models, validation | Already used throughout codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | stdlib | Lightweight result containers | When Pydantic overhead not needed |
| typing.Protocol | stdlib | Pipeline stage interface | For future extensibility |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| functools.lru_cache | cachetools.TTLCache | Use TTL if cache invalidation needed; not needed here |
| sync pipeline | asyncio.gather | Overkill for <1s targets; adds complexity without benefit |
| NetworkX descendants | manual BFS | NetworkX already tested, no reason to hand-roll |

**Installation:**
```bash
# No new dependencies required - all already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
src/skill_retriever/
├── workflows/
│   ├── __init__.py
│   ├── pipeline.py          # RetrievalPipeline coordinator
│   └── models.py             # PipelineResult, ConflictInfo, DependencyChain
├── nodes/retrieval/          # Existing Phase 4 nodes (unchanged)
│   ├── query_planner.py
│   ├── vector_search.py
│   ├── ppr_engine.py
│   ├── flow_pruner.py
│   ├── score_fusion.py
│   └── context_assembler.py
```

### Pattern 1: Coordinator Pipeline
**What:** Single entry point that orchestrates retrieval stages sequentially with early-exit optimization
**When to use:** When stages have dependencies and order matters
**Example:**
```python
# Source: Derived from Phase 4 node composition + RAG pipeline patterns
class RetrievalPipeline:
    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: FAISSVectorStore,
        token_budget: int = 2000,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._budget = token_budget

    def retrieve(self, query: str, component_type: ComponentType | None = None) -> PipelineResult:
        start = time.perf_counter()

        # Stage 1: Plan retrieval strategy
        entities = extract_query_entities(query, self._graph)
        plan = plan_retrieval(query, len(entities))

        # Stage 2: Vector search (always)
        vector_results = search_with_type_filter(
            query, self._vector, self._graph, component_type, plan.max_results
        )

        # Stage 3: Graph retrieval (if plan says so)
        if plan.use_ppr:
            ppr_scores = run_ppr_retrieval(query, self._graph, plan.ppr_alpha)
            if plan.use_flow_pruning:
                paths = flow_based_pruning(ppr_scores, self._graph)
        else:
            ppr_scores = {}

        # Stage 4: Fusion
        fused = fuse_retrieval_results(vector_results, ppr_scores, self._graph, component_type)

        # Stage 5: Resolve dependencies
        with_deps = self._resolve_dependencies(fused)

        # Stage 6: Detect conflicts
        conflicts = self._detect_conflicts(with_deps)

        # Stage 7: Assemble context within budget
        context = assemble_context(with_deps, self._graph, self._budget)

        latency_ms = (time.perf_counter() - start) * 1000
        return PipelineResult(context=context, conflicts=conflicts, latency_ms=latency_ms)
```

### Pattern 2: Transitive Dependency Resolution
**What:** Use NetworkX `descendants()` to find all nodes reachable via DEPENDS_ON edges
**When to use:** When returning component sets that must be complete
**Example:**
```python
# Source: NetworkX DAG algorithms documentation
def resolve_transitive_dependencies(
    component_ids: list[str],
    graph_store: NetworkXGraphStore,
) -> set[str]:
    """Return all components plus their transitive dependencies."""
    all_deps: set[str] = set(component_ids)

    # Build subgraph of only DEPENDS_ON edges
    dep_edges = [
        (e.source_id, e.target_id)
        for cid in component_ids
        for e in graph_store.get_edges(cid)
        if e.edge_type == EdgeType.DEPENDS_ON
    ]

    # Use descendants for transitive closure
    for cid in component_ids:
        descendants = nx.descendants(graph_store._graph, cid)
        all_deps.update(descendants)

    return all_deps
```

### Pattern 3: LRU Query Cache
**What:** Cache full pipeline results keyed by (query, component_type)
**When to use:** Same queries likely to repeat within session
**Example:**
```python
# Source: Python functools documentation
@functools.lru_cache(maxsize=128)
def _cached_retrieve(
    self,
    query: str,
    component_type: str | None,  # Must be hashable, so use str not enum
) -> PipelineResult:
    # Actual retrieval logic
    ...
```

### Anti-Patterns to Avoid
- **Async for <1s operations:** asyncio adds complexity without benefit when total target is 1000ms
- **Caching mutable objects:** `lru_cache` requires hashable args; never cache lists/dicts directly
- **Cycle detection in hot path:** Check for cycles at ingestion time, not retrieval time
- **Unbounded cache:** Always set `maxsize` to prevent memory growth

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Transitive closure | Manual BFS/DFS | `nx.descendants()` | O(V+E), handles edge cases, tested |
| Cycle detection | Manual stack tracking | `nx.is_directed_acyclic_graph()` | Proven algorithm, handles self-loops |
| Query caching | Dict with manual eviction | `@functools.lru_cache` | Thread-safe, O(1), automatic LRU eviction |
| Topological sort | Manual Kahn's algorithm | `nx.topological_sort()` | Raises exception on cycles, handles edge cases |
| High-res timing | `time.time()` | `time.perf_counter()` | Monotonic, higher resolution, no clock drift |

**Key insight:** NetworkX provides all graph algorithms needed. The retrieval nodes from Phase 4 handle all search logic. This phase is pure orchestration and composition.

## Common Pitfalls

### Pitfall 1: Cache Key Unhashability
**What goes wrong:** `@lru_cache` fails because ComponentType enum or list arguments are unhashable
**Why it happens:** Pydantic enums and mutable types can't be dict keys
**How to avoid:** Convert enums to `.value` strings, freeze sets, use tuple for lists
**Warning signs:** `TypeError: unhashable type` at runtime

### Pitfall 2: Token Budget Exceeded After Dependency Resolution
**What goes wrong:** Adding transitive dependencies pushes context over 2000 token limit
**Why it happens:** Dependencies resolved AFTER budget check
**How to avoid:** Resolve dependencies BEFORE calling `assemble_context()`, let assembler truncate
**Warning signs:** Context token count > budget in tests

### Pitfall 3: Missing Latency Tracking
**What goes wrong:** Can't verify 500ms/1000ms SLAs
**Why it happens:** Forgot to instrument timing
**How to avoid:** Wrap entire retrieve() in perf_counter, include latency_ms in result
**Warning signs:** No way to measure compliance with SLAs

### Pitfall 4: Conflict Detection on Wrong Edge Direction
**What goes wrong:** Misses conflicts because checking source->target but conflict stored target->source
**Why it happens:** CONFLICTS_WITH is symmetric but stored directionally
**How to avoid:** Check both incoming and outgoing CONFLICTS_WITH edges
**Warning signs:** Known conflicts not surfaced

### Pitfall 5: Descendants on Node Not in Graph
**What goes wrong:** `nx.descendants()` raises KeyError for missing node
**Why it happens:** Component ID from vector search not yet ingested into graph
**How to avoid:** Check `node in graph` before calling descendants, handle gracefully
**Warning signs:** KeyError in dependency resolution

## Code Examples

Verified patterns from official sources:

### NetworkX Descendants for Transitive Dependencies
```python
# Source: NetworkX 3.5 documentation - dag algorithms
import networkx as nx

def get_all_dependencies(graph: nx.DiGraph, node_id: str) -> set[str]:
    """Return all nodes reachable from node_id (transitive closure)."""
    if node_id not in graph:
        return set()
    return nx.descendants(graph, node_id)
```

### LRU Cache with Statistics
```python
# Source: Python 3.x functools documentation
from functools import lru_cache

@lru_cache(maxsize=128, typed=False)
def cached_search(query: str, type_filter: str | None) -> tuple[str, ...]:
    # Return tuple (immutable) not list
    results = expensive_search(query, type_filter)
    return tuple(r.component_id for r in results)

# Check cache effectiveness
info = cached_search.cache_info()
hit_rate = info.hits / (info.hits + info.misses) if info.misses else 1.0
```

### Latency Monitoring Pattern
```python
# Source: Python time module documentation, PEP 418
import time

def timed_operation() -> tuple[Result, float]:
    """Execute operation and return (result, latency_ms)."""
    start = time.perf_counter()
    result = do_work()
    latency_ms = (time.perf_counter() - start) * 1000
    return result, latency_ms
```

### Conflict Detection via Edge Type
```python
# Source: Existing graph_store.py get_edges() + EdgeType.CONFLICTS_WITH
def detect_conflicts(
    component_ids: set[str],
    graph_store: NetworkXGraphStore,
) -> list[tuple[str, str]]:
    """Find all CONFLICTS_WITH pairs among component_ids."""
    conflicts: list[tuple[str, str]] = []
    checked: set[frozenset[str]] = set()

    for cid in component_ids:
        for edge in graph_store.get_edges(cid):
            if edge.edge_type != EdgeType.CONFLICTS_WITH:
                continue
            other = edge.target_id if edge.source_id == cid else edge.source_id
            if other in component_ids:
                pair = frozenset({cid, other})
                if pair not in checked:
                    conflicts.append((cid, other))
                    checked.add(pair)

    return conflicts
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Async everything | Sync for <1s operations | 2024+ RAG systems | Reduces complexity without latency penalty |
| Manual graph traversal | NetworkX algorithms | NetworkX 3.0+ | Reliable, tested implementations |
| time.time() for perf | time.perf_counter() | PEP 418 (Python 3.3) | Higher resolution, monotonic |
| Global caches | Instance-level caches | Modern patterns | Better testability, isolation |

**Deprecated/outdated:**
- `time.clock()`: Removed in Python 3.8, use `perf_counter()` instead
- `collections.OrderedDict` for LRU: Use `functools.lru_cache` (more efficient)

## Open Questions

Things that couldn't be fully resolved:

1. **Cache Invalidation Strategy**
   - What we know: LRU eviction handles memory; no external data changes during session
   - What's unclear: If components are re-ingested mid-session, cache becomes stale
   - Recommendation: For v1, accept stale cache within session; add `pipeline.clear_cache()` method for manual invalidation

2. **Optimal Cache Size**
   - What we know: 128 is functools default; power of 2 aligns with memory
   - What's unclear: Actual query distribution in production
   - Recommendation: Start with 128, add cache_info() logging, tune in Phase 7

3. **Handling Circular Dependencies**
   - What we know: DEPENDS_ON should form DAG; NetworkX raises on cycles in topological_sort
   - What's unclear: Whether ingestion guarantees no cycles
   - Recommendation: Add cycle check in dependency resolver, return error if cycle detected rather than infinite loop

## Sources

### Primary (HIGH confidence)
- Python functools documentation - lru_cache signature, thread-safety, cache_info()
- NetworkX 3.5 documentation - descendants(), topological_sort(), is_directed_acyclic_graph()
- Python time module documentation - perf_counter() precision and monotonicity
- Existing codebase: `src/skill_retriever/nodes/retrieval/` - all Phase 4 nodes

### Secondary (MEDIUM confidence)
- [RAG Pipeline Orchestration Patterns 2026](https://levelup.gitconnected.com/building-a-scalable-production-grade-agentic-rag-pipeline-1168dcd36260) - layer architecture, caching patterns
- [Super Fast Python - asyncio.gather() timeout](https://superfastpython.com/asyncio-gather-timeout/) - confirmed async not needed for <1s ops
- [Electric Monk Dependency Resolution](https://www.electricmonk.nl/docs/dependency_resolving_algorithm/dependency_resolving_algorithm.html) - topological sort for dependencies

### Tertiary (LOW confidence)
- WebSearch results for cache sizing - validated against functools docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in codebase or stdlib
- Architecture: HIGH - Derived from existing Phase 4 patterns + standard RAG orchestration
- Pitfalls: HIGH - Based on Python documentation and existing code patterns

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (30 days - stable domain)
