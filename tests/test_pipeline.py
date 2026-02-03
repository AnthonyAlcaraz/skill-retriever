"""Tests for RetrievalPipeline coordinator."""

from __future__ import annotations

import numpy as np
import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.memory.vector_store import FAISSVectorStore
from skill_retriever.workflows import PipelineResult, RetrievalPipeline


@pytest.fixture
def graph_store() -> NetworkXGraphStore:
    """Graph store with test components and edges."""
    store = NetworkXGraphStore()

    # Add components of different types
    store.add_node(
        GraphNode(
            id="agent-auth",
            label="AuthenticationAgent",
            component_type=ComponentType.AGENT,
            embedding_id="emb-agent-auth",
        )
    )
    store.add_node(
        GraphNode(
            id="skill-jwt",
            label="JWTTokenSkill",
            component_type=ComponentType.SKILL,
            embedding_id="emb-skill-jwt",
        )
    )
    store.add_node(
        GraphNode(
            id="cmd-login",
            label="LoginCommand",
            component_type=ComponentType.COMMAND,
            embedding_id="emb-cmd-login",
        )
    )
    store.add_node(
        GraphNode(
            id="skill-refresh",
            label="RefreshTokenSkill",
            component_type=ComponentType.SKILL,
            embedding_id="emb-skill-refresh",
        )
    )

    # Add edges to enable PPR traversal
    store.add_edge(
        GraphEdge(
            source_id="agent-auth",
            target_id="skill-jwt",
            edge_type=EdgeType.DEPENDS_ON,
        )
    )
    store.add_edge(
        GraphEdge(
            source_id="skill-jwt",
            target_id="skill-refresh",
            edge_type=EdgeType.ENHANCES,
        )
    )
    store.add_edge(
        GraphEdge(
            source_id="cmd-login",
            target_id="agent-auth",
            edge_type=EdgeType.DEPENDS_ON,
        )
    )

    return store


@pytest.fixture
def vector_store() -> FAISSVectorStore:
    """Vector store with test embeddings."""
    store = FAISSVectorStore(dimensions=384)

    # Create deterministic embeddings for testing
    rng = np.random.default_rng(42)

    # Add embeddings for each component
    store.add("agent-auth", rng.random(384).astype(np.float32))
    store.add("skill-jwt", rng.random(384).astype(np.float32))
    store.add("cmd-login", rng.random(384).astype(np.float32))
    store.add("skill-refresh", rng.random(384).astype(np.float32))

    return store


@pytest.fixture
def pipeline(
    graph_store: NetworkXGraphStore, vector_store: FAISSVectorStore
) -> RetrievalPipeline:
    """RetrievalPipeline with test stores."""
    return RetrievalPipeline(
        graph_store=graph_store,
        vector_store=vector_store,
        token_budget=2000,
        cache_size=128,
    )


class TestPipelineReturnsResult:
    """Test basic retrieval returns PipelineResult."""

    def test_returns_pipeline_result(self, pipeline: RetrievalPipeline) -> None:
        result = pipeline.retrieve("authentication agent")
        assert isinstance(result, PipelineResult)
        assert result.context is not None
        assert isinstance(result.conflicts, list)
        assert isinstance(result.dependencies_added, list)

    def test_returns_components(self, pipeline: RetrievalPipeline) -> None:
        result = pipeline.retrieve("JWT token authentication")
        # Should return at least some components
        assert result.context.components is not None


class TestPipelineCaching:
    """Test caching behavior."""

    def test_second_call_is_cache_hit(self, pipeline: RetrievalPipeline) -> None:
        # First call - cache miss
        result1 = pipeline.retrieve("authentication agent")
        assert not result1.cache_hit

        # Second call - cache hit
        result2 = pipeline.retrieve("authentication agent")
        assert result2.cache_hit

    def test_different_queries_not_cached(self, pipeline: RetrievalPipeline) -> None:
        pipeline.retrieve("authentication agent")
        result = pipeline.retrieve("JWT token skill")
        assert not result.cache_hit

    def test_cache_info_tracks_stats(self, pipeline: RetrievalPipeline) -> None:
        pipeline.retrieve("authentication agent")
        pipeline.retrieve("authentication agent")

        info = pipeline.cache_info
        assert info["hits"] == 1
        assert info["misses"] == 1
        assert info["size"] == 1


class TestPipelineLatency:
    """Test latency tracking."""

    def test_latency_tracked(self, pipeline: RetrievalPipeline) -> None:
        result = pipeline.retrieve("authentication")
        assert result.latency_ms > 0

    def test_cached_call_faster(self, pipeline: RetrievalPipeline) -> None:
        # First call
        result1 = pipeline.retrieve("JWT authentication token")
        first_latency = result1.latency_ms

        # Second call (cached) - should be faster
        result2 = pipeline.retrieve("JWT authentication token")
        cached_latency = result2.latency_ms

        # Cache hit should be significantly faster
        # Note: May occasionally fail on slow systems, but should be rare
        assert cached_latency < first_latency


class TestPipelineTypeFilter:
    """Test type filtering."""

    def test_respects_type_filter(self, pipeline: RetrievalPipeline) -> None:
        result = pipeline.retrieve(
            "authentication", component_type=ComponentType.SKILL
        )
        # All returned components should be skills
        for comp in result.context.components:
            assert "skill" in comp.component_id.lower()


class TestPipelineTokenBudget:
    """Test token budget enforcement."""

    def test_respects_token_budget(
        self, graph_store: NetworkXGraphStore, vector_store: FAISSVectorStore
    ) -> None:
        # Create pipeline with very small budget
        small_budget_pipeline = RetrievalPipeline(
            graph_store=graph_store,
            vector_store=vector_store,
            token_budget=5,  # Very small budget
        )
        result = small_budget_pipeline.retrieve("authentication")
        # Should not exceed budget
        assert result.context.total_tokens <= 5


class TestClearCache:
    """Test cache clearing."""

    def test_clear_cache_resets(self, pipeline: RetrievalPipeline) -> None:
        # Populate cache
        pipeline.retrieve("authentication agent")
        pipeline.retrieve("authentication agent")

        # Clear cache
        pipeline.clear_cache()

        # Cache should be empty
        info = pipeline.cache_info
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["size"] == 0

    def test_after_clear_is_cache_miss(self, pipeline: RetrievalPipeline) -> None:
        pipeline.retrieve("authentication agent")
        pipeline.clear_cache()
        result = pipeline.retrieve("authentication agent")
        assert not result.cache_hit
