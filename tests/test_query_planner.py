"""Tests for query planner and entity extraction."""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.nodes.retrieval import (
    QueryComplexity,
    extract_query_entities,
    plan_retrieval,
)


def _make_node(
    node_id: str, label: str = "", ctype: ComponentType = ComponentType.SKILL,
) -> GraphNode:
    """Helper to build a GraphNode with minimal boilerplate."""
    return GraphNode(id=node_id, component_type=ctype, label=label or node_id)


@pytest.fixture
def graph_with_nodes() -> NetworkXGraphStore:
    """Create a graph store with sample nodes for entity matching."""
    store = NetworkXGraphStore()
    store.add_node(_make_node("skill-agent", "code-agent"))
    store.add_node(_make_node("skill-memory", "memory-manager"))
    store.add_node(_make_node("skill-debug", "debug-tool"))
    store.add_node(_make_node("cmd-git", "git-operations"))
    store.add_node(_make_node("hook-pre", "pre-commit-hook"))
    return store


class TestPlanSimpleQuery:
    """Test 1: plan_retrieval returns SIMPLE for short single-entity queries."""

    def test_short_query_few_entities(self) -> None:
        plan = plan_retrieval("find agent", entity_count=1)

        assert plan.complexity == QueryComplexity.SIMPLE
        assert plan.use_ppr is False
        assert plan.use_flow_pruning is False
        assert plan.ppr_alpha == pytest.approx(0.85)
        assert plan.max_results == 10


class TestPlanModerateQuery:
    """Test 2: plan_retrieval returns MODERATE for medium queries."""

    def test_medium_query_several_entities(self) -> None:
        # 350 chars, 4 entities -> MODERATE
        query = "x" * 350
        plan = plan_retrieval(query, entity_count=4)

        assert plan.complexity == QueryComplexity.MODERATE
        assert plan.use_ppr is True
        assert plan.use_flow_pruning is False
        assert plan.ppr_alpha == pytest.approx(0.85)
        assert plan.max_results == 20


class TestPlanComplexQuery:
    """Test 3: plan_retrieval returns COMPLEX for long multi-entity queries."""

    def test_long_query_many_entities(self) -> None:
        # 700 chars, 6 entities -> COMPLEX (either condition triggers it)
        query = "a" * 700
        plan = plan_retrieval(query, entity_count=6)

        assert plan.complexity == QueryComplexity.COMPLEX
        assert plan.use_ppr is True
        assert plan.use_flow_pruning is True
        assert plan.ppr_alpha == pytest.approx(0.7)
        assert plan.max_results == 30

    def test_long_query_alone_triggers_complex(self) -> None:
        # 650 chars but only 2 entities -> still COMPLEX due to length
        query = "b" * 650
        plan = plan_retrieval(query, entity_count=2)

        assert plan.complexity == QueryComplexity.COMPLEX

    def test_many_entities_alone_triggers_complex(self) -> None:
        # Short query but 7 entities -> COMPLEX
        plan = plan_retrieval("short", entity_count=7)

        assert plan.complexity == QueryComplexity.COMPLEX


class TestExtractFiltersStopwords:
    """Test 4: extract_query_entities filters stopwords."""

    def test_stopwords_filtered(self, graph_with_nodes: NetworkXGraphStore) -> None:
        # "how to use the agent" should only pass "agent" token
        matched = extract_query_entities("how to use the agent", graph_with_nodes)

        # "agent" matches node with label "code-agent"
        assert "skill-agent" in matched
        # No matches for stopwords
        assert len(matched) == 1


class TestExtractMatchesCaseInsensitive:
    """Test 5: extract_query_entities matches graph node labels case-insensitively."""

    def test_case_insensitive_matching(
        self, graph_with_nodes: NetworkXGraphStore,
    ) -> None:
        # "Agent" (capitalized) should match "code-agent" label
        matched = extract_query_entities("Agent", graph_with_nodes)

        assert "skill-agent" in matched

    def test_uppercase_query(self, graph_with_nodes: NetworkXGraphStore) -> None:
        # "MEMORY" should match "memory-manager"
        matched = extract_query_entities("MEMORY", graph_with_nodes)

        assert "skill-memory" in matched


class TestExtractReturnsEmptyNoMatches:
    """Test 6: extract_query_entities returns empty set when no matches."""

    def test_no_matches(self, graph_with_nodes: NetworkXGraphStore) -> None:
        # Query with no matching tokens
        matched = extract_query_entities("zebra unicorn", graph_with_nodes)

        assert matched == set()

    def test_only_stopwords(self, graph_with_nodes: NetworkXGraphStore) -> None:
        # Query with only stopwords
        matched = extract_query_entities("the is a to", graph_with_nodes)

        assert matched == set()

    def test_empty_graph(self) -> None:
        # Empty graph should return empty set
        empty_store = NetworkXGraphStore()
        matched = extract_query_entities("agent memory", empty_store)

        assert matched == set()
