"""Tests for flow-based graph pruning."""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.nodes.retrieval.flow_pruner import (
    compute_path_reliability,
    find_paths_between,
    flow_based_pruning,
)


def _make_node(
    node_id: str,
    label: str = "",
    ctype: ComponentType = ComponentType.SKILL,
) -> GraphNode:
    """Helper to build a GraphNode with minimal boilerplate."""
    return GraphNode(id=node_id, component_type=ctype, label=label or node_id)


def _make_edge(source: str, target: str, weight: float = 1.0) -> GraphEdge:
    """Helper to build a GraphEdge."""
    return GraphEdge(
        source_id=source,
        target_id=target,
        edge_type=EdgeType.DEPENDS_ON,
        weight=weight,
    )


@pytest.fixture
def simple_graph() -> NetworkXGraphStore:
    """Create a simple graph for path finding tests."""
    store = NetworkXGraphStore()
    # A -> B -> C -> D
    store.add_node(_make_node("A", "node-a"))
    store.add_node(_make_node("B", "node-b"))
    store.add_node(_make_node("C", "node-c"))
    store.add_node(_make_node("D", "node-d"))

    store.add_edge(_make_edge("A", "B"))
    store.add_edge(_make_edge("B", "C"))
    store.add_edge(_make_edge("C", "D"))
    store.add_edge(_make_edge("A", "C"))  # Shortcut

    return store


@pytest.fixture
def large_graph() -> NetworkXGraphStore:
    """Create a larger graph for reduction testing (20 nodes, many edges)."""
    store = NetworkXGraphStore()

    # Create 20 nodes
    for i in range(20):
        store.add_node(_make_node(f"node-{i:02d}", f"label-{i:02d}"))

    # Create dense connectivity - hub-and-spoke plus chain
    # Hub nodes: 0, 5, 10, 15
    hubs = [0, 5, 10, 15]
    for hub in hubs:
        for i in range(20):
            if i != hub and i not in hubs:
                store.add_edge(_make_edge(f"node-{hub:02d}", f"node-{i:02d}"))
                store.add_edge(_make_edge(f"node-{i:02d}", f"node-{hub:02d}"))

    # Chain between hubs
    store.add_edge(_make_edge("node-00", "node-05"))
    store.add_edge(_make_edge("node-05", "node-10"))
    store.add_edge(_make_edge("node-10", "node-15"))
    store.add_edge(_make_edge("node-05", "node-00"))  # Bidirectional
    store.add_edge(_make_edge("node-10", "node-05"))
    store.add_edge(_make_edge("node-15", "node-10"))

    return store


class TestPathReliabilityAverage:
    """Test 1: compute_path_reliability returns average of PPR scores."""

    def test_average_calculation(self) -> None:
        ppr_scores = {"A": 0.3, "B": 0.2, "C": 0.1}
        path = ["A", "B", "C"]

        reliability = compute_path_reliability(path, ppr_scores)

        # Average of 0.3, 0.2, 0.1 = 0.2
        assert reliability == pytest.approx(0.2)

    def test_missing_nodes_treated_as_zero(self) -> None:
        ppr_scores = {"A": 0.3, "C": 0.3}
        path = ["A", "B", "C"]

        reliability = compute_path_reliability(path, ppr_scores)

        # Average of 0.3, 0.0, 0.3 = 0.2
        assert reliability == pytest.approx(0.2)


class TestPathReliabilityEmpty:
    """Test 2: compute_path_reliability returns 0.0 for empty path."""

    def test_empty_path(self) -> None:
        ppr_scores = {"A": 0.3, "B": 0.2}

        reliability = compute_path_reliability([], ppr_scores)

        assert reliability == 0.0


class TestFindPathsBetween:
    """Test 3: find_paths_between finds simple paths."""

    def test_finds_direct_path(self, simple_graph: NetworkXGraphStore) -> None:
        paths = find_paths_between("A", "B", simple_graph._graph)

        assert len(paths) >= 1
        assert ["A", "B"] in paths

    def test_finds_multiple_paths(self, simple_graph: NetworkXGraphStore) -> None:
        # A->D can go A->B->C->D or A->C->D
        paths = find_paths_between("A", "D", simple_graph._graph)

        assert len(paths) >= 2
        # At least one longer and one shorter path
        path_lengths = [len(p) for p in paths]
        assert min(path_lengths) < max(path_lengths)


class TestFindPathsRespectsMaxLength:
    """Test 4: find_paths_between respects max_length constraint."""

    def test_max_length_filtering(self, simple_graph: NetworkXGraphStore) -> None:
        # With max_length=2, should only find A->C->D (2 edges), not A->B->C->D (3 edges)
        paths = find_paths_between("A", "D", simple_graph._graph, max_length=2)

        for path in paths:
            # Path length is number of nodes, edges = nodes - 1
            assert len(path) - 1 <= 2

    def test_no_paths_if_too_short(self, simple_graph: NetworkXGraphStore) -> None:
        # B->D requires at least 2 hops (B->C->D), max_length=1 should fail
        paths = find_paths_between("B", "D", simple_graph._graph, max_length=1)

        assert len(paths) == 0


class TestFlowPruningSorted:
    """Test 5: flow_based_pruning returns paths sorted by reliability."""

    def test_sorted_by_reliability(self, simple_graph: NetworkXGraphStore) -> None:
        # Create PPR scores favoring certain nodes
        ppr_scores = {
            "A": 0.4,
            "B": 0.3,
            "C": 0.2,
            "D": 0.1,
        }

        paths = flow_based_pruning(
            ppr_scores, simple_graph, threshold=0.001, max_paths=10
        )

        if len(paths) >= 2:
            # Verify descending order by reliability
            for i in range(len(paths) - 1):
                assert paths[i].reliability >= paths[i + 1].reliability


class TestFlowPruningRespectsMaxPaths:
    """Test 6: flow_based_pruning respects max_paths limit."""

    def test_max_paths_limit(self, simple_graph: NetworkXGraphStore) -> None:
        ppr_scores = {
            "A": 0.4,
            "B": 0.3,
            "C": 0.2,
            "D": 0.1,
        }

        paths = flow_based_pruning(
            ppr_scores, simple_graph, threshold=0.001, max_paths=2
        )

        assert len(paths) <= 2


class TestFlowPruningFiltersLowReliability:
    """Test 7: flow_based_pruning filters paths below threshold."""

    def test_threshold_filtering(self, simple_graph: NetworkXGraphStore) -> None:
        # Low scores - most paths will be filtered
        ppr_scores = {
            "A": 0.05,
            "B": 0.01,
            "C": 0.01,
            "D": 0.01,
        }

        # High threshold should filter many paths
        paths_high_thresh = flow_based_pruning(
            ppr_scores, simple_graph, threshold=0.1, max_paths=50
        )
        paths_low_thresh = flow_based_pruning(
            ppr_scores, simple_graph, threshold=0.001, max_paths=50
        )

        # High threshold should have fewer or equal paths
        assert len(paths_high_thresh) <= len(paths_low_thresh)

        # All returned paths should meet threshold
        for path in paths_high_thresh:
            assert path.reliability >= 0.1


class TestFlowPruningReduction:
    """Test 8: flow_based_pruning achieves 40%+ reduction (CRITICAL)."""

    def test_forty_percent_reduction(self, large_graph: NetworkXGraphStore) -> None:
        """Flow pruning should reduce unique nodes by at least 40% vs raw PPR.

        This validates the core benefit of path-based pruning: focusing on
        structurally important nodes rather than all high-scoring nodes.
        """
        # Simulate PPR scores for many nodes (say 15 out of 20)
        # In real usage, these come from run_ppr_retrieval
        ppr_scores = {
            f"node-{i:02d}": 0.1 - (i * 0.005)  # Decreasing scores
            for i in range(15)
        }

        # Run flow pruning with max_paths=10
        paths = flow_based_pruning(
            ppr_scores,
            large_graph,
            threshold=0.01,
            max_paths=10,
            max_endpoints=8,
        )

        # Count unique nodes in pruned paths
        unique_pruned_nodes: set[str] = set()
        for path in paths:
            unique_pruned_nodes.update(path.nodes)

        # Count nodes with significant PPR score (> 0.01)
        ppr_nodes_count = len([s for s in ppr_scores.values() if s > 0.01])

        # Assert 40%+ reduction: pruned nodes <= 60% of PPR nodes
        # This means we've removed at least 40% of low-value nodes
        if ppr_nodes_count > 0 and len(paths) > 0:
            reduction_ratio = len(unique_pruned_nodes) / ppr_nodes_count
            assert reduction_ratio <= 0.6, (
                f"Expected 40%+ reduction, got {(1 - reduction_ratio) * 100:.1f}% reduction. "
                f"Pruned: {len(unique_pruned_nodes)}, PPR: {ppr_nodes_count}"
            )

    def test_empty_ppr_returns_empty(self, large_graph: NetworkXGraphStore) -> None:
        """Empty PPR scores should return empty paths."""
        paths = flow_based_pruning({}, large_graph)

        assert paths == []
