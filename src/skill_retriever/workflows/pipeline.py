"""RetrievalPipeline coordinator with caching and latency monitoring."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skill_retriever.nodes.retrieval.context_assembler import (
    RetrievalContext,
    assemble_context,
)
from skill_retriever.nodes.retrieval.flow_pruner import flow_based_pruning
from skill_retriever.nodes.retrieval.models import RankedComponent
from skill_retriever.nodes.retrieval.ppr_engine import run_ppr_retrieval
from skill_retriever.nodes.retrieval.query_planner import (
    extract_query_entities,
    plan_retrieval,
)
from skill_retriever.nodes.retrieval.external_search import (
    external_results_to_ranked,
    search_skillsh,
)
from skill_retriever.nodes.retrieval.score_fusion import fuse_retrieval_results
from skill_retriever.nodes.retrieval.vector_search import search_with_type_filter
from skill_retriever.workflows.dependency_resolver import (
    detect_conflicts,
    resolve_transitive_dependencies,
)
from skill_retriever.workflows.models import ConflictInfo, PipelineResult

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentType
    from skill_retriever.memory.component_memory import ComponentMemory
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.memory.vector_store import FAISSVectorStore


# Minimum score for dependencies added via transitive resolution
DEPENDENCY_MIN_SCORE = 0.1

HIGH_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_CACHE_SIZE = 128
DEFAULT_TOKEN_BUDGET = 2000


@dataclass(frozen=True)
class _CachedResult:
    """Internal cached result containing context and resolution metadata."""

    context: RetrievalContext
    dependencies_added: tuple[str, ...]
    conflicts: tuple[ConflictInfo, ...]
    # RETR-06: Abstraction level awareness
    abstraction_level: str = "medium"
    suggested_types: tuple[str, ...] = ()


class RetrievalPipeline:
    """Coordinates retrieval stages with caching and latency monitoring.

    Orchestrates query planning, vector search, PPR, flow pruning,
    score fusion, and context assembly into a single entry point.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: FAISSVectorStore,
        component_memory: ComponentMemory | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """Initialize the pipeline with stores and configuration.

        Args:
            graph_store: Graph store for PPR and component lookup.
            vector_store: FAISS vector store for semantic search.
            component_memory: Optional component memory for usage-based ranking (LRNG-04).
            token_budget: Maximum tokens for context assembly.
            cache_size: LRU cache size for query results.
        """
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._component_memory = component_memory
        self._token_budget = token_budget
        self._cache_size = cache_size

        # Track cache stats manually since we can't access lru_cache info
        # from a bound method's wrapper easily
        self._cache_hits = 0
        self._cache_misses = 0

        # Create cached retrieval function
        self._retrieve_cached = functools.lru_cache(maxsize=cache_size)(
            self._retrieve_impl
        )

    def _retrieve_impl(
        self,
        query: str,
        component_type_str: str | None,
        top_k: int,
    ) -> _CachedResult:
        """Internal implementation of retrieval (cacheable).

        Args:
            query: Natural language search query.
            component_type_str: Component type as string (for hashability) or None.
            top_k: Maximum number of results.

        Returns:
            _CachedResult with context, dependencies_added, and conflicts.
        """
        # Convert type string back to enum if provided
        component_type: ComponentType | None = None
        if component_type_str is not None:
            from skill_retriever.entities.components import ComponentType

            component_type = ComponentType(component_type_str)

        # Stage 1: Query planning
        entities = extract_query_entities(query, self._graph_store)
        plan = plan_retrieval(query, len(entities))

        # Stage 2: Vector search (always runs)
        vector_results = search_with_type_filter(
            query,
            self._vector_store,
            self._graph_store,
            component_type=None,  # Type filter applied after fusion
            top_k=plan.max_results,
        )

        # Stage 2.5: External API search (skills.sh) - non-blocking
        external_ranked: list[str] | None = None
        external_skill_map: dict[str, object] = {}
        try:
            external_skills = search_skillsh(query, limit=20)
            if external_skills:
                all_node_ids = self._graph_store.get_all_node_ids()
                external_ranked, external_skill_map = external_results_to_ranked(
                    external_skills, all_node_ids
                )
        except Exception:
            pass  # External search failure never blocks local results

        # Early exit optimization: skip graph if not needed and high confidence
        skip_graph = (
            not plan.use_ppr
            and vector_results
            and vector_results[0].score > HIGH_CONFIDENCE_THRESHOLD
        )

        graph_results: dict[str, float] = {}
        if not skip_graph and plan.use_ppr:
            # Stage 3: PPR and optional flow pruning
            ppr_results = run_ppr_retrieval(
                query,
                self._graph_store,
                alpha=plan.ppr_alpha,
                top_k=plan.max_results,
            )

            if plan.use_flow_pruning and ppr_results:
                # Run flow pruning for path extraction
                paths = flow_based_pruning(
                    ppr_results,
                    self._graph_store,
                )
                # Add path nodes to PPR results with boosted scores
                for path in paths:
                    for node_id in path.nodes:
                        if node_id not in ppr_results:
                            # Add with reliability score
                            ppr_results[node_id] = path.reliability

            graph_results = ppr_results

        # Stage 4: Score fusion with usage-based boosting (LRNG-04)
        fused_results = fuse_retrieval_results(
            vector_results,
            graph_results,
            self._graph_store,
            component_memory=self._component_memory,
            component_type=component_type,
            top_k=top_k,
            external_ranked=external_ranked,
        )

        # Stage 5a: Resolve transitive dependencies BEFORE context assembly
        fused_ids = [comp.component_id for comp in fused_results]
        all_component_ids, dependencies_added = resolve_transitive_dependencies(
            fused_ids, self._graph_store
        )

        # Stage 5b: Detect conflicts among all components (fused + deps)
        conflicts = detect_conflicts(all_component_ids, self._graph_store)

        # Stage 5c: Add dependency components to results for context assembly
        # Dependencies get a minimum score so they appear in context
        expanded_results = list(fused_results)
        next_rank = len(fused_results) + 1
        for dep_id in dependencies_added:
            expanded_results.append(
                RankedComponent(
                    component_id=dep_id,
                    score=DEPENDENCY_MIN_SCORE,
                    rank=next_rank,
                    source="dependency",
                )
            )
            next_rank += 1

        # Stage 6: Context assembly with token budget
        context = assemble_context(
            expanded_results,
            self._graph_store,
            token_budget=self._token_budget,
        )

        return _CachedResult(
            context=context,
            dependencies_added=tuple(dependencies_added),
            conflicts=tuple(conflicts),
            # RETR-06: Pass through abstraction level from query plan
            abstraction_level=plan.abstraction_level.value,
            suggested_types=tuple(plan.suggested_types),
        )

    def retrieve(
        self,
        query: str,
        component_type: ComponentType | None = None,
        top_k: int = 10,
    ) -> PipelineResult:
        """Execute full retrieval pipeline, returning cached result if available.

        Args:
            query: Natural language search query.
            component_type: Optional type filter for results.
            top_k: Maximum number of results.

        Returns:
            PipelineResult with context, conflicts, latency, and cache status.
        """
        # Convert component_type to string for cache key hashability
        type_str = component_type.value if component_type is not None else None

        # Check cache info before call
        cache_info_before = self._retrieve_cached.cache_info()

        # Time the retrieval
        start = time.perf_counter()
        cached_result = self._retrieve_cached(query, type_str, top_k)
        end = time.perf_counter()

        # Check cache info after call
        cache_info_after = self._retrieve_cached.cache_info()
        cache_hit = cache_info_after.hits > cache_info_before.hits

        # Update internal stats
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        latency_ms = (end - start) * 1000

        return PipelineResult(
            context=cached_result.context,
            conflicts=list(cached_result.conflicts),
            dependencies_added=list(cached_result.dependencies_added),
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            # RETR-06: Abstraction level awareness
            abstraction_level=cached_result.abstraction_level,
            suggested_types=list(cached_result.suggested_types),
        )

    def clear_cache(self) -> None:
        """Invalidate all cached results."""
        self._retrieve_cached.cache_clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_info(self) -> dict[str, int]:
        """Return cache statistics (hits, misses, size).

        Returns:
            Dictionary with 'hits', 'misses', 'size', and 'maxsize' keys.
        """
        info = self._retrieve_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize or 0,
        }
