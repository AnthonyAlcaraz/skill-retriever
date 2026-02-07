"""Tests for FalkorDB hybrid graph store.

Offline tests (no Docker required): test write-through to NetworkX,
PPR, Protocol compliance, fallback behavior.

Online tests (require FalkorDB running): marked with @pytest.mark.docker.
"""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.memory.falkordb_store import FalkorDBGraphStore
from skill_retriever.memory.graph_store import GraphStore


def _make_node(
    node_id: str, label: str = "", ctype: ComponentType = ComponentType.SKILL,
) -> GraphNode:
    return GraphNode(id=node_id, component_type=ctype, label=label or node_id)


def _make_edge(
    src: str,
    tgt: str,
    etype: EdgeType = EdgeType.DEPENDS_ON,
    weight: float = 1.0,
) -> GraphEdge:
    return GraphEdge(source_id=src, target_id=tgt, edge_type=etype, weight=weight)


class TestOfflineProtocolCompliance:
    """FalkorDBGraphStore satisfies GraphStore protocol in offline mode."""

    def test_isinstance(self) -> None:
        store = FalkorDBGraphStore()
        assert isinstance(store, GraphStore)


class TestOfflineAddAndGet:
    """Offline mode: add_node / get_node / add_edge / get_edges work via NetworkX."""

    def test_add_and_retrieve_node(self) -> None:
        store = FalkorDBGraphStore()
        node = _make_node("a", "Alpha", ComponentType.AGENT)
        store.add_node(node)

        result = store.get_node("a")
        assert result is not None
        assert result.id == "a"
        assert result.component_type == ComponentType.AGENT
        assert result.label == "Alpha"

    def test_get_nonexistent_returns_none(self) -> None:
        store = FalkorDBGraphStore()
        assert store.get_node("missing") is None

    def test_add_and_retrieve_edge(self) -> None:
        store = FalkorDBGraphStore()
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


class TestOfflineNeighbors:
    """Offline mode: get_neighbors returns both predecessors and successors."""

    def test_bidirectional_neighbors(self) -> None:
        store = FalkorDBGraphStore()
        for nid in ("a", "b", "c"):
            store.add_node(_make_node(nid))
        store.add_edge(_make_edge("a", "b"))
        store.add_edge(_make_edge("b", "c"))

        neighbors = store.get_neighbors("b")
        neighbor_ids = {n.id for n in neighbors}
        assert neighbor_ids == {"a", "c"}


class TestOfflineCounts:
    """Offline mode: node_count and edge_count."""

    def test_counts(self) -> None:
        store = FalkorDBGraphStore()
        assert store.node_count() == 0
        assert store.edge_count() == 0

        store.add_node(_make_node("a"))
        store.add_node(_make_node("b"))
        store.add_edge(_make_edge("a", "b"))

        assert store.node_count() == 2
        assert store.edge_count() == 1


class TestOfflinePPR:
    """Offline mode: PPR uses NetworkX mirror correctly."""

    def test_ppr_chain(self) -> None:
        store = FalkorDBGraphStore()
        for nid in ("a", "b", "c", "d"):
            store.add_node(_make_node(nid))
        store.add_edge(_make_edge("a", "b"))
        store.add_edge(_make_edge("b", "c"))
        store.add_edge(_make_edge("c", "d"))

        results = store.personalized_pagerank(seed_ids=["a"], top_k=10)
        result_ids = [nid for nid, _ in results]

        assert "a" not in result_ids
        assert result_ids.index("b") < result_ids.index("d")

    def test_ppr_empty(self) -> None:
        store = FalkorDBGraphStore()
        assert store.personalized_pagerank(seed_ids=["a"]) == []

    def test_ppr_empty_seeds(self) -> None:
        store = FalkorDBGraphStore()
        store.add_node(_make_node("a"))
        assert store.personalized_pagerank(seed_ids=[]) == []


class TestOfflineSaveLoad:
    """Offline mode: save/load round-trip via JSON."""

    def test_round_trip(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        filepath = str(tmp_path / "graph.json")

        store = FalkorDBGraphStore()
        store.add_node(_make_node("x", "ExNode", ComponentType.MCP))
        store.add_node(_make_node("y", "WhyNode", ComponentType.HOOK))
        store.add_edge(_make_edge("x", "y", EdgeType.BUNDLES_WITH, weight=3.0))
        store.save(filepath)

        loaded = FalkorDBGraphStore()
        loaded.load(filepath)

        assert loaded.node_count() == 2
        assert loaded.edge_count() == 1

        node_x = loaded.get_node("x")
        assert node_x is not None
        assert node_x.label == "ExNode"


class TestOfflineNxGraph:
    """Offline mode: nx_graph property returns the internal graph."""

    def test_nx_graph_property(self) -> None:
        store = FalkorDBGraphStore()
        store.add_node(_make_node("a"))
        store.add_node(_make_node("b"))
        store.add_edge(_make_edge("a", "b"))

        graph = store.nx_graph
        assert "a" in graph
        assert "b" in graph
        assert graph.has_edge("a", "b")


class TestOfflineLabelIndex:
    """Offline mode: get_label_index returns correct mapping."""

    def test_label_index(self) -> None:
        store = FalkorDBGraphStore()
        store.add_node(_make_node("a", "Git Helper"))
        store.add_node(_make_node("b", "Docker Setup"))

        index = store.get_label_index()
        assert "git helper" in index
        assert "docker setup" in index
        assert index["git helper"] == ["a"]


class TestOfflineDependsOnSubgraph:
    """Offline mode: get_depends_on_subgraph filters correctly."""

    def test_depends_on_subgraph(self) -> None:
        store = FalkorDBGraphStore()
        store.add_node(_make_node("a"))
        store.add_node(_make_node("b"))
        store.add_node(_make_node("c"))
        store.add_edge(_make_edge("a", "b", EdgeType.DEPENDS_ON))
        store.add_edge(_make_edge("b", "c", EdgeType.ENHANCES))

        subgraph = store.get_depends_on_subgraph()
        assert subgraph.has_edge("a", "b")
        assert not subgraph.has_edge("b", "c")


class TestOfflineBatchOps:
    """Offline mode: batch operations write to NetworkX."""

    def test_add_nodes_batch(self) -> None:
        store = FalkorDBGraphStore()
        nodes = [_make_node(f"n{i}") for i in range(10)]
        count = store.add_nodes_batch(nodes)

        assert count == 10
        assert store.node_count() == 10

    def test_add_edges_batch(self) -> None:
        store = FalkorDBGraphStore()
        nodes = [_make_node(f"n{i}") for i in range(5)]
        store.add_nodes_batch(nodes)

        edges = [_make_edge(f"n{i}", f"n{i+1}") for i in range(4)]
        count = store.add_edges_batch(edges)

        assert count == 4
        assert store.edge_count() == 4


class TestFallbackBehavior:
    """Test graceful fallback when FalkorDB is unavailable."""

    def test_offline_store_works_without_connection(self) -> None:
        store = FalkorDBGraphStore(connection=None)
        store.add_node(_make_node("test"))
        assert store.node_count() == 1
        assert store.get_node("test") is not None

    def test_sync_without_connection_is_noop(self) -> None:
        store = FalkorDBGraphStore(connection=None)
        store.sync_from_falkordb()
        assert store.node_count() == 0
