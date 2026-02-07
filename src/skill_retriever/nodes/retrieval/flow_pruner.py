"""Flow-based pruning for graph retrieval path extraction (PathRAG-style)."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from skill_retriever.memory.graph_store import GraphStore

FLOW_CONFIG = {
    "alpha": 0.85,
    "threshold": 0.01,
    "max_path_length": 4,
    "max_paths": 10,
    "max_endpoints": 8,
}


@dataclass
class RetrievalPath:
    """A path extracted from the graph with flow and reliability metrics."""

    nodes: list[str]
    flow: float
    reliability: float


def compute_path_reliability(path: list[str], ppr_scores: dict[str, float]) -> float:
    """Calculate average PPR score of nodes in path.

    Args:
        path: List of node IDs in the path
        ppr_scores: Dictionary mapping node IDs to PPR scores

    Returns:
        Average PPR score across path nodes, or 0.0 for empty paths.
    """
    if not path:
        return 0.0
    scores = [ppr_scores.get(node, 0.0) for node in path]
    return sum(scores) / len(scores)


def find_paths_between(
    source: str,
    target: str,
    graph: nx.DiGraph[str],  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    max_length: int = 4,
) -> list[list[str]]:
    """Find all simple paths between source and target up to max_length using BFS.

    Args:
        source: Source node ID
        target: Target node ID
        graph: NetworkX directed graph
        max_length: Maximum path length (edge count)

    Returns:
        List of paths, where each path is a list of node IDs.
    """
    if source not in graph or target not in graph:
        return []

    paths: list[list[str]] = []
    try:
        # Use networkx all_simple_paths with cutoff
        for path in nx.all_simple_paths(graph, source, target, cutoff=max_length):  # pyright: ignore[reportUnknownArgumentType]
            paths.append(list(path))
            if len(paths) >= 5:  # Limit to prevent explosion
                break
    except nx.NetworkXNoPath:
        pass
    return paths


def flow_based_pruning(
    ppr_scores: dict[str, float],
    graph_store: GraphStore,
    threshold: float = 0.01,
    max_paths: int = 10,
    max_endpoints: int = 8,
) -> list[RetrievalPath]:
    """Extract key relational paths using flow propagation (PathRAG-style).

    Selects top PPR-scored nodes as endpoints, then finds all simple paths
    between them, filtering by reliability threshold.

    Args:
        ppr_scores: Dictionary mapping node IDs to PPR scores
        graph_store: Graph store to search for paths
        threshold: Minimum average PPR score for path inclusion
        max_paths: Maximum number of paths to return
        max_endpoints: Maximum number of endpoint nodes to consider

    Returns:
        List of RetrievalPath objects sorted by reliability (descending).
    """
    if not ppr_scores:
        return []

    graph = graph_store.nx_graph

    # Get top endpoints from PPR scores
    endpoints = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[
        :max_endpoints
    ]

    all_paths: list[RetrievalPath] = []

    for i, (src, _) in enumerate(endpoints):
        for tgt, _ in endpoints[i + 1 :]:
            # Find paths in both directions
            paths = find_paths_between(src, tgt, graph, int(FLOW_CONFIG["max_path_length"]))
            paths.extend(
                find_paths_between(tgt, src, graph, int(FLOW_CONFIG["max_path_length"]))
            )

            for path_nodes in paths:
                reliability = compute_path_reliability(path_nodes, ppr_scores)
                if reliability >= threshold:
                    # Compute flow as product of edge weights (default 1.0)
                    flow = 1.0
                    for j in range(len(path_nodes) - 1):
                        edge_data = graph.get_edge_data(path_nodes[j], path_nodes[j + 1])
                        if edge_data:
                            flow *= edge_data.get("weight", 1.0)

                    all_paths.append(
                        RetrievalPath(
                            nodes=path_nodes,
                            flow=flow,
                            reliability=reliability,
                        )
                    )

    # Sort by reliability descending and return top max_paths
    all_paths.sort(key=lambda p: p.reliability, reverse=True)
    return all_paths[:max_paths]
