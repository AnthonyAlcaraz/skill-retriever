"""RRF score fusion for hybrid retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skill_retriever.nodes.retrieval.models import RankedComponent

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentType
    from skill_retriever.memory.graph_store import GraphStore

RRF_K = 60  # Empirically validated default from Elasticsearch/Milvus


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


def fuse_retrieval_results(
    vector_results: list[RankedComponent],
    graph_results: dict[str, float],
    graph_store: GraphStore,
    component_type: ComponentType | None = None,
    top_k: int = 10,
) -> list[RankedComponent]:
    """Fuse vector and graph retrieval results using RRF.

    Type filter applied AFTER fusion (not during retrieval) per research.
    """
    # Extract ranked list from vector_results
    vector_ranked = [r.component_id for r in vector_results]

    # Extract ranked list from graph_results (sort by score descending)
    graph_ranked = [
        item_id
        for item_id, _ in sorted(graph_results.items(), key=lambda x: x[1], reverse=True)
    ]

    # Handle empty graph results (vector-only fallback)
    if graph_ranked:
        fused = reciprocal_rank_fusion([vector_ranked, graph_ranked])
    else:
        fused = reciprocal_rank_fusion([vector_ranked])

    # Apply type filter AFTER fusion
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
