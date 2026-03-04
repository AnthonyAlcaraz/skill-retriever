"""Post-crawl graph edge enrichment for auto-populating relationships.

Adds SAME_CATEGORY and ENHANCES edges between related components
based on category grouping and tool overlap analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from skill_retriever.entities.graph import EdgeType, GraphEdge

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentMetadata
    from skill_retriever.memory.graph_store import GraphStore


def enrich_graph_edges(components: list[ComponentMetadata], graph_store: GraphStore) -> int:
    """Add SAME_CATEGORY and ENHANCES edges between related components.

    Args:
        components: List of components to analyze for relationships.
        graph_store: Graph store to add edges to.

    Returns:
        Count of edges added.
    """
    edges_added = 0

    # Group by category
    by_category: dict[str, list[ComponentMetadata]] = defaultdict(list)
    for comp in components:
        if comp.category:
            by_category[comp.category].append(comp)

    # SAME_CATEGORY edges (within same category, same repo)
    for _category, comps in by_category.items():
        for i, a in enumerate(comps):
            for b in comps[i + 1 :]:
                if a.source_repo == b.source_repo:
                    graph_store.add_edge(
                        GraphEdge(
                            source_id=a.id,
                            target_id=b.id,
                            edge_type=EdgeType.SAME_CATEGORY,
                            weight=0.5,
                        )
                    )
                    edges_added += 1

    # ENHANCES edges (tool overlap between different component types)
    for i, a in enumerate(components):
        if not a.tools:
            continue
        for b in components[i + 1 :]:
            if not b.tools:
                continue
            overlap = set(a.tools) & set(b.tools)
            if overlap and a.component_type != b.component_type:
                weight = len(overlap) / max(len(a.tools), len(b.tools))
                if weight >= 0.3:
                    graph_store.add_edge(
                        GraphEdge(
                            source_id=a.id,
                            target_id=b.id,
                            edge_type=EdgeType.ENHANCES,
                            weight=weight,
                        )
                    )
                    edges_added += 1

    return edges_added
