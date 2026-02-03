"""Tests for RRF score fusion."""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.nodes.retrieval.models import RankedComponent
from skill_retriever.nodes.retrieval.score_fusion import (
    fuse_retrieval_results,
    reciprocal_rank_fusion,
)


class TestRRFTopItemFirst:
    """Test that items appearing high in both lists rank first."""

    def test_top_item_appears_first(self) -> None:
        list1 = ["a", "b", "c"]
        list2 = ["a", "c", "b"]
        result = reciprocal_rank_fusion([list1, list2])
        # 'a' is rank 1 in both lists, should be first
        assert result[0][0] == "a"


class TestRRFSingleListItems:
    """Test items appearing in only one list."""

    def test_handles_single_list_items(self) -> None:
        list1 = ["a", "b"]
        list2 = ["c", "d"]
        result = reciprocal_rank_fusion([list1, list2])
        # All items should appear
        result_ids = [item[0] for item in result]
        assert set(result_ids) == {"a", "b", "c", "d"}


class TestRRFKParameter:
    """Test k parameter affects score distribution."""

    def test_k_affects_distribution(self) -> None:
        list1 = ["a", "b"]
        result_k1 = reciprocal_rank_fusion([list1], k=1)
        result_k60 = reciprocal_rank_fusion([list1], k=60)
        # With k=1: scores are 1/2 and 1/3
        # With k=60: scores are 1/61 and 1/62
        # Score gap is larger with k=1
        gap_k1 = result_k1[0][1] - result_k1[1][1]
        gap_k60 = result_k60[0][1] - result_k60[1][1]
        assert gap_k1 > gap_k60


@pytest.fixture
def fusion_graph() -> NetworkXGraphStore:
    """Graph with components for fusion testing."""
    store = NetworkXGraphStore()
    store.add_node(GraphNode(id="agent-1", label="TestAgent", component_type=ComponentType.AGENT))
    store.add_node(GraphNode(id="skill-1", label="TestSkill", component_type=ComponentType.SKILL))
    store.add_node(GraphNode(id="cmd-1", label="TestCmd", component_type=ComponentType.COMMAND))
    return store


class TestFuseResults:
    """Test fuse_retrieval_results combines vector and graph."""

    def test_combines_results(self, fusion_graph: NetworkXGraphStore) -> None:
        vector = [
            RankedComponent(component_id="agent-1", score=0.9, rank=1, source="vector"),
            RankedComponent(component_id="skill-1", score=0.8, rank=2, source="vector"),
        ]
        graph = {"skill-1": 0.5, "agent-1": 0.3}
        result = fuse_retrieval_results(vector, graph, fusion_graph)
        assert len(result) >= 1
        assert all(r.source == "fused" for r in result)


class TestFuseTypeFilter:
    """Test type filter applied after fusion."""

    def test_filters_by_type(self, fusion_graph: NetworkXGraphStore) -> None:
        vector = [
            RankedComponent(component_id="agent-1", score=0.9, rank=1, source="vector"),
            RankedComponent(component_id="skill-1", score=0.8, rank=2, source="vector"),
        ]
        graph = {"skill-1": 0.5, "agent-1": 0.3}
        result = fuse_retrieval_results(
            vector, graph, fusion_graph, component_type=ComponentType.SKILL
        )
        assert all(r.component_id.startswith("skill") for r in result)


class TestFuseSource:
    """Test fused results have source='fused'."""

    def test_source_is_fused(self, fusion_graph: NetworkXGraphStore) -> None:
        vector = [
            RankedComponent(component_id="agent-1", score=0.9, rank=1, source="vector"),
        ]
        graph = {"agent-1": 0.3}
        result = fuse_retrieval_results(vector, graph, fusion_graph)
        assert result[0].source == "fused"


class TestFuseTopK:
    """Test fuse respects top_k limit."""

    def test_respects_top_k(self, fusion_graph: NetworkXGraphStore) -> None:
        vector = [
            RankedComponent(component_id="agent-1", score=0.9, rank=1, source="vector"),
            RankedComponent(component_id="skill-1", score=0.8, rank=2, source="vector"),
            RankedComponent(component_id="cmd-1", score=0.7, rank=3, source="vector"),
        ]
        graph = {"agent-1": 0.5, "skill-1": 0.4, "cmd-1": 0.3}
        result = fuse_retrieval_results(vector, graph, fusion_graph, top_k=2)
        assert len(result) == 2


class TestFuseEmptyGraph:
    """Test vector-only fallback when graph results empty."""

    def test_handles_empty_graph(self, fusion_graph: NetworkXGraphStore) -> None:
        vector = [
            RankedComponent(component_id="agent-1", score=0.9, rank=1, source="vector"),
        ]
        graph: dict[str, float] = {}
        result = fuse_retrieval_results(vector, graph, fusion_graph)
        assert len(result) == 1
        assert result[0].source == "fused"
