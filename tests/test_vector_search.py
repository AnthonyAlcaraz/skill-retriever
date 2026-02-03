"""Tests for vector search node with text embedding and type filtering."""

from __future__ import annotations

import numpy as np
import pytest
from fastembed import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]

from skill_retriever.config import EMBEDDING_CONFIG
from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.memory.vector_store import FAISSVectorStore
from skill_retriever.nodes.retrieval import (
    search_by_text,
    search_with_type_filter,
)


def _make_node(
    node_id: str, label: str = "", ctype: ComponentType = ComponentType.SKILL,
) -> GraphNode:
    """Helper to build a GraphNode with minimal boilerplate."""
    return GraphNode(id=node_id, component_type=ctype, label=label or node_id)


@pytest.fixture(scope="module")
def embedding_model() -> TextEmbedding:
    """Create a shared embedding model for the test module."""
    return TextEmbedding(
        model_name=EMBEDDING_CONFIG.model_name,
        cache_dir=EMBEDDING_CONFIG.cache_dir,
    )


@pytest.fixture
def populated_stores(embedding_model: TextEmbedding) -> tuple[FAISSVectorStore, NetworkXGraphStore]:
    """Create vector and graph stores with sample components."""
    vector_store = FAISSVectorStore()
    graph_store = NetworkXGraphStore()

    # Sample components with different types
    components = [
        ("comp-agent-1", "Code generation agent", ComponentType.AGENT),
        ("comp-agent-2", "Testing automation agent", ComponentType.AGENT),
        ("comp-skill-1", "Git operations skill", ComponentType.SKILL),
        ("comp-skill-2", "File search skill", ComponentType.SKILL),
        ("comp-mcp-1", "Database MCP server", ComponentType.MCP),
    ]

    # Add to graph store and generate embeddings
    texts = []
    ids = []
    for comp_id, desc, ctype in components:
        graph_store.add_node(_make_node(comp_id, desc, ctype))
        texts.append(desc)
        ids.append(comp_id)

    # Generate embeddings in batch
    embeddings = list(embedding_model.embed(texts))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    embeddings_array = np.array(embeddings, dtype=np.float32)  # pyright: ignore[reportUnknownArgumentType]
    vector_store.add_batch(ids, embeddings_array)

    return vector_store, graph_store


class TestSearchByTextReturnsRanked:
    """Test 1: search_by_text returns RankedComponent with source='vector'."""

    def test_returns_ranked_component(
        self, populated_stores: tuple[FAISSVectorStore, NetworkXGraphStore],
    ) -> None:
        vector_store, _ = populated_stores

        results = search_by_text("agent", vector_store, top_k=3)

        assert len(results) > 0
        for result in results:
            assert result.source == "vector"
            assert result.rank >= 1
            assert result.component_id.startswith("comp-")


class TestSearchByTextSortedByScore:
    """Test 2: search_by_text returns results sorted by score descending."""

    def test_sorted_descending(
        self, populated_stores: tuple[FAISSVectorStore, NetworkXGraphStore],
    ) -> None:
        vector_store, _ = populated_stores

        results = search_by_text("code generation", vector_store, top_k=5)

        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Ranks should be sequential 1, 2, 3, ...
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))


class TestSearchWithTypeFilterFilters:
    """Test 3: search_with_type_filter returns only requested type."""

    def test_filters_by_type(
        self, populated_stores: tuple[FAISSVectorStore, NetworkXGraphStore],
    ) -> None:
        vector_store, graph_store = populated_stores

        results = search_with_type_filter(
            "operations",
            vector_store,
            graph_store,
            component_type=ComponentType.SKILL,
            top_k=5,
        )

        # All results should be SKILL type
        for result in results:
            node = graph_store.get_node(result.component_id)
            assert node is not None
            assert node.component_type == ComponentType.SKILL

    def test_reranks_after_filter(
        self, populated_stores: tuple[FAISSVectorStore, NetworkXGraphStore],
    ) -> None:
        vector_store, graph_store = populated_stores

        results = search_with_type_filter(
            "agent",
            vector_store,
            graph_store,
            component_type=ComponentType.AGENT,
            top_k=5,
        )

        # Ranks should be sequential 1, 2, 3, ... after filtering
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))


class TestSearchWithTypeFilterNoneReturnsAll:
    """Test 4: search_with_type_filter with None type returns all types."""

    def test_none_type_returns_all(
        self, populated_stores: tuple[FAISSVectorStore, NetworkXGraphStore],
    ) -> None:
        vector_store, graph_store = populated_stores

        results = search_with_type_filter(
            "operations",
            vector_store,
            graph_store,
            component_type=None,
            top_k=5,
        )

        # Should return results without type filtering
        assert len(results) == 5  # All 5 components

        # Should include multiple types
        types = set()
        for result in results:
            node = graph_store.get_node(result.component_id)
            if node:
                types.add(node.component_type)

        # At least 2 different types present
        assert len(types) >= 2


class TestSearchEmptyStore:
    """Test 5: search on empty store returns empty list."""

    def test_empty_vector_store(self) -> None:
        empty_store = FAISSVectorStore()

        results = search_by_text("anything", empty_store, top_k=10)

        assert results == []

    def test_empty_with_type_filter(self) -> None:
        empty_vector = FAISSVectorStore()
        empty_graph = NetworkXGraphStore()

        results = search_with_type_filter(
            "anything",
            empty_vector,
            empty_graph,
            component_type=ComponentType.AGENT,
            top_k=10,
        )

        assert results == []
