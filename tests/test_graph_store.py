"""Tests for graph store protocol and NetworkX implementation."""

from __future__ import annotations

import random
import time

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.memory.graph_store import GraphStore, NetworkXGraphStore


def _make_node(
    node_id: str, label: str = "", ctype: ComponentType = ComponentType.SKILL,
) -> GraphNode:
    """Helper to build a GraphNode with minimal boilerplate."""
    return GraphNode(id=node_id, component_type=ctype, label=label or node_id)


def _make_edge(
    src: str,
    tgt: str,
    etype: EdgeType = EdgeType.DEPENDS_ON,
    weight: float = 1.0,
) -> GraphEdge:
    """Helper to build a GraphEdge with minimal boilerplate."""
    return GraphEdge(source_id=src, target_id=tgt, edge_type=etype, weight=weight)


class TestAddAndGetNode:
    """Test 1: add_node / get_node round-trip and None for missing."""

    def test_add_and_retrieve(self) -> None:
        store = NetworkXGraphStore()
        node = _make_node("a", "Alpha", ComponentType.AGENT)
        store.add_node(node)

        result = store.get_node("a")
        assert result is not None
        assert result.id == "a"
        assert result.component_type == ComponentType.AGENT
        assert result.label == "Alpha"

    def test_get_nonexistent_returns_none(self) -> None:
        store = NetworkXGraphStore()
        assert store.get_node("missing") is None


class TestAddAndGetEdges:
    """Test 2: add_edge / get_edges round-trip."""

    def test_add_and_retrieve_edge(self) -> None:
        store = NetworkXGraphStore()
        store.add_node(_make_node("a"))
        store.add_node(_make_node("b"))
        edge = _make_edge("a", "b", EdgeType.ENHANCES, weight=2.5)
        store.add_edge(edge)

        edges = store.get_edges("a")
        assert len(edges) == 1
        assert edges[0].source_id == "a"
        assert edges[0].target_id == "b"
        assert edges[0].edge_type == EdgeType.ENHANCES
        assert edges[0].weight == pytest.approx(2.5)


class TestGetNeighbors:
    """Test 3: get_neighbors returns both predecessors and successors."""

    def test_bidirectional_neighbors(self) -> None:
        store = NetworkXGraphStore()
        for nid in ("a", "b", "c"):
            store.add_node(_make_node(nid))
        store.add_edge(_make_edge("a", "b"))
        store.add_edge(_make_edge("b", "c"))

        neighbors = store.get_neighbors("b")
        neighbor_ids = {n.id for n in neighbors}
        assert neighbor_ids == {"a", "c"}


class TestNodeAndEdgeCounts:
    """Test 4: node_count and edge_count."""

    def test_counts(self) -> None:
        store = NetworkXGraphStore()
        assert store.node_count() == 0
        assert store.edge_count() == 0

        store.add_node(_make_node("a"))
        store.add_node(_make_node("b"))
        store.add_edge(_make_edge("a", "b"))

        assert store.node_count() == 2
        assert store.edge_count() == 1


class TestPersonalizedPageRank:
    """Test 5: PPR basic ranking on a chain graph."""

    def test_ppr_chain(self) -> None:
        store = NetworkXGraphStore()
        for nid in ("a", "b", "c", "d"):
            store.add_node(_make_node(nid))
        store.add_edge(_make_edge("a", "b"))
        store.add_edge(_make_edge("b", "c"))
        store.add_edge(_make_edge("c", "d"))

        results = store.personalized_pagerank(seed_ids=["a"], top_k=10)
        result_ids = [nid for nid, _ in results]

        # Seed 'a' must be excluded
        assert "a" not in result_ids
        # B should rank higher than D (closer to seed)
        assert result_ids.index("b") < result_ids.index("d")


class TestPPREmptyGraph:
    """Test 6: PPR on empty graph returns []."""

    def test_empty(self) -> None:
        store = NetworkXGraphStore()
        assert store.personalized_pagerank(seed_ids=["a"]) == []


class TestPPREmptySeeds:
    """Test 7: PPR with empty seeds returns []."""

    def test_empty_seeds(self) -> None:
        store = NetworkXGraphStore()
        store.add_node(_make_node("a"))
        assert store.personalized_pagerank(seed_ids=[]) == []


class TestSaveAndLoad:
    """Test 8: save/load round-trip preserves graph data."""

    def test_round_trip(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        filepath = str(tmp_path / "graph.json")

        store = NetworkXGraphStore()
        store.add_node(_make_node("x", "ExNode", ComponentType.MCP))
        store.add_node(_make_node("y", "WhyNode", ComponentType.HOOK))
        store.add_edge(_make_edge("x", "y", EdgeType.BUNDLES_WITH, weight=3.0))
        store.save(filepath)

        loaded = NetworkXGraphStore()
        loaded.load(filepath)

        assert loaded.node_count() == 2
        assert loaded.edge_count() == 1

        node_x = loaded.get_node("x")
        assert node_x is not None
        assert node_x.label == "ExNode"
        assert node_x.component_type == ComponentType.MCP

        edges = loaded.get_edges("x")
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(3.0)


class TestPPRPerformance:
    """Test 9: PPR on 1300-node graph completes in < 200ms."""

    def test_ppr_under_200ms(self) -> None:
        store = NetworkXGraphStore()
        rng = random.Random(42)
        node_ids = [f"n{i}" for i in range(1300)]
        for nid in node_ids:
            store.add_node(_make_node(nid))
        # Add ~5000 random edges
        for _ in range(5000):
            src = rng.choice(node_ids)
            tgt = rng.choice(node_ids)
            if src != tgt:
                store.add_edge(_make_edge(src, tgt))

        start = time.perf_counter()
        results = store.personalized_pagerank(seed_ids=["n0"], top_k=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"PPR took {elapsed_ms:.1f}ms, expected < 200ms"
        assert len(results) <= 10


class TestProtocolIsinstance:
    """Test 10: NetworkXGraphStore satisfies GraphStore protocol."""

    def test_isinstance(self) -> None:
        store = NetworkXGraphStore()
        assert isinstance(store, GraphStore)
