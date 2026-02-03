"""Tests for PPR engine with adaptive alpha."""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.nodes.retrieval.ppr_engine import (
    PPR_CONFIG,
    compute_adaptive_alpha,
    run_ppr_retrieval,
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
def graph_with_gsd_nodes() -> NetworkXGraphStore:
    """Create a graph store with GSD-related nodes for entity matching."""
    store = NetworkXGraphStore()
    # Add nodes with labels for entity matching
    store.add_node(_make_node("node-gsd", "gsd"))
    store.add_node(_make_node("node-command", "command"))
    store.add_node(_make_node("node-agent", "agent"))
    store.add_node(_make_node("node-tool", "tool"))
    store.add_node(_make_node("node-workflow", "workflow"))
    store.add_node(_make_node("node-memory", "memory"))
    store.add_node(_make_node("node-graph", "graph"))
    store.add_node(_make_node("node-vector", "vector"))
    store.add_node(_make_node("node-retrieval", "retrieval"))
    store.add_node(_make_node("node-search", "search"))

    # Add edges to create graph structure
    store.add_edge(_make_edge("node-gsd", "node-command"))
    store.add_edge(_make_edge("node-gsd", "node-agent"))
    store.add_edge(_make_edge("node-command", "node-tool"))
    store.add_edge(_make_edge("node-agent", "node-memory"))
    store.add_edge(_make_edge("node-agent", "node-workflow"))
    store.add_edge(_make_edge("node-memory", "node-graph"))
    store.add_edge(_make_edge("node-memory", "node-vector"))
    store.add_edge(_make_edge("node-retrieval", "node-search"))
    store.add_edge(_make_edge("node-retrieval", "node-vector"))
    store.add_edge(_make_edge("node-retrieval", "node-graph"))

    return store


class TestAdaptiveAlphaSpecific:
    """Test 1: compute_adaptive_alpha returns 0.9 for specific queries."""

    def test_named_entity_narrow_scope(self) -> None:
        # "Find GSD command" has named entity (GSD), 2 seeds -> specific
        alpha = compute_adaptive_alpha("Find GSD command", seed_count=2)

        assert alpha == pytest.approx(0.9)

    def test_pascal_case_entity(self) -> None:
        # PascalCase triggers named entity detection
        alpha = compute_adaptive_alpha("Use GraphStore", seed_count=1)

        assert alpha == pytest.approx(0.9)


class TestAdaptiveAlphaBroad:
    """Test 2: compute_adaptive_alpha returns 0.6 for broad queries."""

    def test_many_seeds(self) -> None:
        # Broad query with 7 seeds -> broad alpha
        alpha = compute_adaptive_alpha("search across memory graph vector tools", seed_count=7)

        assert alpha == pytest.approx(0.6)

    def test_six_seeds_is_broad(self) -> None:
        # 6 seeds crosses the >5 threshold
        alpha = compute_adaptive_alpha("some query", seed_count=6)

        assert alpha == pytest.approx(0.6)


class TestAdaptiveAlphaDefault:
    """Test 3: compute_adaptive_alpha returns 0.85 for moderate queries."""

    def test_moderate_seeds(self) -> None:
        # Medium query with 4 seeds, no named entities -> default
        alpha = compute_adaptive_alpha("find all related tools", seed_count=4)

        assert alpha == pytest.approx(0.85)

    def test_no_named_entity_narrow(self) -> None:
        # Narrow (2 seeds) but no named entity -> default (not specific)
        alpha = compute_adaptive_alpha("get tools", seed_count=2)

        assert alpha == pytest.approx(0.85)


class TestPPREmptySeedsReturnsEmpty:
    """Test 4: run_ppr_retrieval returns empty dict when no matching seeds."""

    def test_no_matching_seeds(self, graph_with_gsd_nodes: NetworkXGraphStore) -> None:
        # Query with no matching entities
        result = run_ppr_retrieval("zebra unicorn elephant", graph_with_gsd_nodes)

        assert result == {}

    def test_empty_graph(self) -> None:
        # Empty graph should return empty
        empty_store = NetworkXGraphStore()
        result = run_ppr_retrieval("gsd command", empty_store)

        assert result == {}


class TestPPRReturnsScores:
    """Test 5: run_ppr_retrieval returns non-empty dict with scores."""

    def test_valid_seeds_returns_scores(
        self,
        graph_with_gsd_nodes: NetworkXGraphStore,
    ) -> None:
        # "gsd command" matches nodes -> should return PPR scores
        result = run_ppr_retrieval("gsd command", graph_with_gsd_nodes)

        assert len(result) > 0
        # All values should be positive scores
        assert all(score > 0 for score in result.values())


class TestPPRFiltersLowScores:
    """Test 6: run_ppr_retrieval filters scores below minimum."""

    def test_scores_above_minimum(
        self,
        graph_with_gsd_nodes: NetworkXGraphStore,
    ) -> None:
        # All returned scores should be >= min_score
        result = run_ppr_retrieval("gsd agent memory", graph_with_gsd_nodes)
        min_score = PPR_CONFIG["min_score"]

        assert all(score >= min_score for score in result.values())


class TestPPRUsesProvidedAlpha:
    """Test 7: run_ppr_retrieval uses explicit alpha when provided."""

    def test_explicit_alpha_overrides(
        self,
        graph_with_gsd_nodes: NetworkXGraphStore,
    ) -> None:
        # Run with explicit alpha=0.5
        result_custom = run_ppr_retrieval(
            "gsd command",
            graph_with_gsd_nodes,
            alpha=0.5,
            top_k=10,
        )

        # Run with default adaptive alpha
        result_adaptive = run_ppr_retrieval(
            "gsd command",
            graph_with_gsd_nodes,
            alpha=None,
            top_k=10,
        )

        # Both should return results, but scores will differ due to alpha
        assert len(result_custom) > 0
        assert len(result_adaptive) > 0
        # Scores should differ (different alpha values)
        # We can't guarantee order, but score distribution should differ
        custom_scores = sorted(result_custom.values(), reverse=True)
        adaptive_scores = sorted(result_adaptive.values(), reverse=True)
        # At least one score should be different
        assert custom_scores != adaptive_scores or len(result_custom) != len(result_adaptive)
