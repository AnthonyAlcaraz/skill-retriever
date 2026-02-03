"""Personalized PageRank engine with adaptive alpha for graph-enhanced retrieval."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from skill_retriever.nodes.retrieval.query_planner import extract_query_entities

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore

PPR_CONFIG = {
    "default_alpha": 0.85,
    "specific_alpha": 0.9,
    "broad_alpha": 0.6,
    "default_top_k": 50,
    "min_score": 0.001,
}


def compute_adaptive_alpha(query: str, seed_count: int) -> float:
    """Return PPR alpha based on query characteristics.

    Alpha tuning:
    - Specific (0.9): Named entity + narrow scope -> stay close to seeds
    - Broad (0.6): Many seeds -> explore more broadly
    - Default (0.85): Balanced exploration/exploitation
    """
    # Detect named entities (PascalCase, camelCase, or capitalized words)
    has_named_entity = bool(re.search(r"\b[A-Z][a-z]+\w*\b", query))
    is_narrow = seed_count <= 3

    if has_named_entity and is_narrow:
        return PPR_CONFIG["specific_alpha"]  # 0.9
    elif seed_count > 5:
        return PPR_CONFIG["broad_alpha"]  # 0.6
    return PPR_CONFIG["default_alpha"]  # 0.85


def run_ppr_retrieval(
    query: str,
    graph_store: GraphStore,
    alpha: float | None = None,
    top_k: int = 50,
) -> dict[str, float]:
    """Run PPR with adaptive alpha and seed extraction.

    Args:
        query: Search query text
        graph_store: Graph store to run PPR against
        alpha: PPR damping factor (if None, computed adaptively)
        top_k: Maximum number of results

    Returns:
        Dictionary mapping node IDs to their PPR scores, filtered by min_score.
        Returns empty dict if no seeds found (fallback to vector-only).
    """
    seeds = extract_query_entities(query, graph_store)

    if not seeds:
        return {}  # Fall back to vector-only

    if alpha is None:
        alpha = compute_adaptive_alpha(query, len(seeds))

    results = graph_store.personalized_pagerank(
        seed_ids=list(seeds),
        alpha=alpha,
        top_k=top_k,
    )

    # Filter by minimum score
    min_score = PPR_CONFIG["min_score"]
    return {node_id: score for node_id, score in results if score >= min_score}
