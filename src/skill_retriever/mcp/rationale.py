"""Path-to-rationale conversion for component recommendations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skill_retriever.entities.graph import EdgeType

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.nodes.retrieval.context_assembler import RetrievalContext
    from skill_retriever.nodes.retrieval.models import RankedComponent

# Human-readable descriptions for edge types
EDGE_DESCRIPTIONS: dict[EdgeType, str] = {
    EdgeType.DEPENDS_ON: "requires",
    EdgeType.ENHANCES: "enhances",
    EdgeType.CONFLICTS_WITH: "conflicts with",
    EdgeType.BUNDLES_WITH: "bundles with",
    EdgeType.SAME_CATEGORY: "in same category as",
}


def path_to_explanation(path_nodes: list[str], graph_store: GraphStore) -> str:
    """Convert a graph path to a human-readable explanation.

    Args:
        path_nodes: List of node IDs representing the path.
        graph_store: Graph store for edge lookup.

    Returns:
        Human-readable explanation of the path, or fallback text.
    """
    if len(path_nodes) < 2:
        return "Direct match"

    segments: list[str] = []
    for i in range(len(path_nodes) - 1):
        source_id = path_nodes[i]
        target_id = path_nodes[i + 1]

        # Get node labels for readable output
        source_node = graph_store.get_node(source_id)
        target_node = graph_store.get_node(target_id)

        source_label = source_node.label if source_node else source_id
        target_label = target_node.label if target_node else target_id

        # Find edge between source and target
        edges = graph_store.get_edges(source_id)
        verb = "relates to"  # Default fallback
        for edge in edges:
            if edge.target_id == target_id:
                verb = EDGE_DESCRIPTIONS.get(edge.edge_type, "relates to")
                break
            if edge.source_id == target_id:
                # Incoming edge from target
                verb = EDGE_DESCRIPTIONS.get(edge.edge_type, "relates to")
                break

        segments.append(f"{source_label} {verb} {target_label}")

    if not segments:
        return "Graph traversal"

    return " -> ".join(segments)


def generate_rationale(
    component: RankedComponent,
    context: RetrievalContext,
    graph_store: GraphStore,
) -> str:
    """Generate a rationale for a component recommendation.

    Args:
        component: The ranked component to explain.
        context: The retrieval context containing all results.
        graph_store: Graph store for relationship lookup.

    Returns:
        Human-readable rationale under 50 words.
    """
    source = component.source

    # Handle different retrieval sources
    if source == "vector":
        return "Semantic match to query"

    if source == "dependency":
        # Find what this component is a dependency of
        for other in context.components:
            if other.component_id == component.component_id:
                continue
            edges = graph_store.get_edges(other.component_id)
            for edge in edges:
                if (
                    edge.edge_type == EdgeType.DEPENDS_ON
                    and edge.target_id == component.component_id
                ):
                    other_node = graph_store.get_node(other.component_id)
                    parent_label = other_node.label if other_node else other.component_id
                    return f"Required dependency of {parent_label}"
        return "Required dependency"

    if source in ("graph", "fused", "hybrid"):
        # Get node info for better explanation
        node = graph_store.get_node(component.component_id)
        if node:
            neighbors = graph_store.get_neighbors(component.component_id)
            if neighbors:
                # Find strongest relationship
                edges = graph_store.get_edges(component.component_id)
                for edge in edges:
                    edge_desc = EDGE_DESCRIPTIONS.get(edge.edge_type)
                    if edge_desc:
                        other_id = (
                            edge.target_id
                            if edge.source_id == component.component_id
                            else edge.source_id
                        )
                        other_node = graph_store.get_node(other_id)
                        other_label = other_node.label if other_node else other_id
                        return f"Graph: {node.label} {edge_desc} {other_label}"
            return f"Graph match: {node.label}"
        return "Graph traversal match"

    # Fallback for unknown sources
    return "Matched query"
