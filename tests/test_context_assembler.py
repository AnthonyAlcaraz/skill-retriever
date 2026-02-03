"""Tests for context assembler with token budgeting."""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.nodes.retrieval.context_assembler import (
    assemble_context,
    estimate_tokens,
)
from skill_retriever.nodes.retrieval.models import RankedComponent


class TestEstimateTokens:
    """Test token estimation."""

    def test_returns_approximately_len_div_4(self) -> None:
        text = "a" * 100
        assert estimate_tokens(text) == 25


@pytest.fixture
def context_graph() -> NetworkXGraphStore:
    """Graph with components for context testing."""
    store = NetworkXGraphStore()
    # Agent with short label (10 chars = ~2 tokens)
    store.add_node(GraphNode(id="agent-1", label="TestAgent1", component_type=ComponentType.AGENT))
    # Skill with short label
    store.add_node(GraphNode(id="skill-1", label="TestSkill1", component_type=ComponentType.SKILL))
    # Command with short label
    store.add_node(GraphNode(id="cmd-1", label="TestCmd123", component_type=ComponentType.COMMAND))
    # Agent with long label (100 chars = ~25 tokens)
    store.add_node(GraphNode(id="agent-2", label="A" * 100, component_type=ComponentType.AGENT))
    return store


class TestAssembleRespectsBudget:
    """Test context assembly respects token budget."""

    def test_respects_budget(self, context_graph: NetworkXGraphStore) -> None:
        ranked = [
            RankedComponent(component_id="agent-2", score=0.9, rank=1, source="fused"),
            RankedComponent(component_id="skill-1", score=0.8, rank=2, source="fused"),
        ]
        # Budget of 10 tokens - only short components fit
        result = assemble_context(ranked, context_graph, token_budget=10)
        # agent-2 has 25 tokens, skill-1 has ~2 tokens
        assert result.truncated
        assert result.total_tokens <= 10


class TestAssemblePrioritizesType:
    """Test agents prioritized over skills over commands."""

    def test_agents_before_skills(self, context_graph: NetworkXGraphStore) -> None:
        ranked = [
            RankedComponent(component_id="skill-1", score=0.9, rank=1, source="fused"),
            RankedComponent(component_id="agent-1", score=0.8, rank=2, source="fused"),
        ]
        result = assemble_context(ranked, context_graph, token_budget=1000)
        # Agent should come first despite lower score
        assert result.components[0].component_id == "agent-1"


class TestAssembleTruncatedFlag:
    """Test truncated flag when budget exceeded."""

    def test_truncated_when_exceeded(self, context_graph: NetworkXGraphStore) -> None:
        ranked = [
            RankedComponent(component_id="agent-2", score=0.9, rank=1, source="fused"),
        ]
        # agent-2 has 25 tokens, budget is 5
        result = assemble_context(ranked, context_graph, token_budget=5)
        assert result.truncated
        assert len(result.components) == 0


class TestAssembleExcludedCount:
    """Test excluded_count tracking."""

    def test_tracks_excluded_count(self, context_graph: NetworkXGraphStore) -> None:
        ranked = [
            RankedComponent(component_id="agent-1", score=0.9, rank=1, source="fused"),
            RankedComponent(component_id="skill-1", score=0.8, rank=2, source="fused"),
            RankedComponent(component_id="cmd-1", score=0.7, rank=3, source="fused"),
        ]
        # Budget allows 2 short components (~4 tokens total)
        result = assemble_context(ranked, context_graph, token_budget=6)
        assert result.excluded_count == len(ranked) - len(result.components)


class TestAssembleAllWithinBudget:
    """Test truncated=False when all fit."""

    def test_not_truncated_when_all_fit(self, context_graph: NetworkXGraphStore) -> None:
        ranked = [
            RankedComponent(component_id="agent-1", score=0.9, rank=1, source="fused"),
        ]
        result = assemble_context(ranked, context_graph, token_budget=1000)
        assert not result.truncated
        assert result.excluded_count == 0


class TestAssembleEmpty:
    """Test empty input handling."""

    def test_handles_empty_input(self, context_graph: NetworkXGraphStore) -> None:
        result = assemble_context([], context_graph)
        assert result.components == []
        assert result.total_tokens == 0
        assert not result.truncated
        assert result.excluded_count == 0
