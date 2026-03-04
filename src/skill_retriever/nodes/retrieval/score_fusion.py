"""RRF score fusion for hybrid retrieval with usage-based boosting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skill_retriever.nodes.retrieval.models import RankedComponent

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentType
    from skill_retriever.memory.component_memory import ComponentMemory
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.memory.metadata_store import MetadataStore
    from skill_retriever.memory.outcome_tracker import OutcomeTracker

RRF_K = 60  # Empirically validated default from Elasticsearch/Milvus

# Usage-based boosting parameters (LRNG-04)
SELECTION_BOOST_MAX = 0.5  # Max 50% boost for highly selected components
CO_SELECTION_BOOST = 0.1  # 10% boost per co-selected component in results


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    For each ranked list, compute 1/(k + rank) score for each item (rank starts at 1).
    Sum scores across all lists for each item.
    Return sorted list of (item_id, rrf_score) tuples, descending by score.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank_idx, item_id in enumerate(ranked_list):
            rank = rank_idx + 1  # rank starts at 1
            rrf_score = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score

    # Sort by score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


def _apply_usage_boost(
    fused_scores: list[tuple[str, float]],
    component_memory: ComponentMemory | None,
) -> list[tuple[str, float]]:
    """Apply usage-based boosting to fused scores (LRNG-04).

    Boosts components based on:
    1. Selection rate (how often selected when recommended)
    2. Co-selection patterns (components frequently selected together)

    Args:
        fused_scores: List of (component_id, rrf_score) tuples.
        component_memory: Component memory for usage stats (may be None).

    Returns:
        List of (component_id, boosted_score) tuples, re-sorted by score.
    """
    if component_memory is None:
        return fused_scores

    # Get all component IDs in this result set for co-selection lookup
    result_ids = {item_id for item_id, _ in fused_scores}

    boosted: list[tuple[str, float]] = []
    for item_id, score in fused_scores:
        boost_factor = 1.0

        # Boost 1: Selection rate (max +50%)
        selection_rate = component_memory.get_selection_rate(item_id)
        boost_factor += SELECTION_BOOST_MAX * selection_rate

        # Boost 2: Co-selection with other results (max +30%)
        co_selected = component_memory.get_co_selected(item_id, top_k=5)
        co_boost = 0.0
        for other_id, count in co_selected:
            if other_id in result_ids and count >= 2:
                # Stronger boost for frequently co-selected pairs
                co_boost += CO_SELECTION_BOOST * min(count / 5, 1.0)
        boost_factor += min(co_boost, 0.3)  # Cap at 30%

        boosted.append((item_id, score * boost_factor))

    # Re-sort by boosted score
    return sorted(boosted, key=lambda x: x[1], reverse=True)


def _apply_quality_adjustments(
    fused_scores: list[tuple[str, float]],
    outcome_tracker: OutcomeTracker | None,
    metadata_store: MetadataStore | None,
) -> list[tuple[str, float]]:
    """Apply success rate and deprecation adjustments to fused scores.

    - Success rate multiplier: score *= 0.7 + 0.3 * success_rate
    - Deprecation penalty: score *= 0.1 if deprecated_at is set

    Args:
        fused_scores: List of (component_id, score) tuples.
        outcome_tracker: Outcome tracker for success rates (may be None).
        metadata_store: Metadata store for deprecation status (may be None).

    Returns:
        List of (component_id, adjusted_score) tuples, re-sorted by score.
    """
    if outcome_tracker is None and metadata_store is None:
        return fused_scores

    adjusted: list[tuple[str, float]] = []
    for item_id, score in fused_scores:
        multiplier = 1.0

        # Success rate multiplier (SkillRL)
        if outcome_tracker is not None:
            stats = outcome_tracker.outcomes.get(item_id)
            if stats is not None:
                total = stats.install_successes + stats.install_failures
                if total > 0:
                    multiplier *= 0.7 + 0.3 * stats.success_rate

        # Deprecation penalty (SkillRL)
        if metadata_store is not None:
            meta = metadata_store.get(item_id)
            if meta is not None and meta.deprecated_at is not None:
                multiplier *= 0.1

        adjusted.append((item_id, score * multiplier))

    return sorted(adjusted, key=lambda x: x[1], reverse=True)


def fuse_retrieval_results(
    vector_results: list[RankedComponent],
    graph_results: dict[str, float],
    graph_store: GraphStore,
    component_memory: ComponentMemory | None = None,
    component_type: ComponentType | None = None,
    top_k: int = 10,
    external_ranked: list[str] | None = None,
    outcome_tracker: OutcomeTracker | None = None,
    metadata_store: MetadataStore | None = None,
) -> list[RankedComponent]:
    """Fuse vector, graph, and external retrieval results using RRF with usage boosting.

    Type filter applied AFTER fusion (not during retrieval) per research.
    Usage-based boosting applied when component_memory is provided (LRNG-04).
    Quality adjustments (success rate, deprecation) applied when trackers provided (SkillRL).

    Args:
        vector_results: Results from vector search.
        graph_results: Results from PPR graph traversal.
        graph_store: Graph store for node lookup.
        component_memory: Optional component memory for usage-based boosting.
        component_type: Optional type filter.
        top_k: Maximum results to return.
        external_ranked: Optional ranked list from external API (skills.sh).
        outcome_tracker: Optional outcome tracker for success rate adjustments.
        metadata_store: Optional metadata store for deprecation penalty.

    Returns:
        List of RankedComponent with fused and boosted scores.
    """
    # Extract ranked list from vector_results
    vector_ranked = [r.component_id for r in vector_results]

    # Extract ranked list from graph_results (sort by score descending)
    graph_ranked = [
        item_id
        for item_id, _ in sorted(graph_results.items(), key=lambda x: x[1], reverse=True)
    ]

    # Build list of ranked lists for RRF fusion
    ranked_lists = [vector_ranked]
    if graph_ranked:
        ranked_lists.append(graph_ranked)
    if external_ranked:
        ranked_lists.append(external_ranked)
    fused = reciprocal_rank_fusion(ranked_lists)

    # Apply usage-based boosting (LRNG-04)
    fused = _apply_usage_boost(fused, component_memory)

    # Apply quality adjustments: success rate + deprecation (SkillRL)
    fused = _apply_quality_adjustments(fused, outcome_tracker, metadata_store)

    # Apply type filter AFTER fusion and boosting
    results: list[RankedComponent] = []
    for item_id, score in fused:
        if component_type is not None:
            node = graph_store.get_node(item_id)
            if node is None or node.component_type != component_type:
                continue

        results.append(
            RankedComponent(
                component_id=item_id,
                score=score,
                rank=len(results) + 1,
                source="fused",
            )
        )

        if len(results) >= top_k:
            break

    return results
