"""Pytest fixtures for validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from ranx import Qrels

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.memory.vector_store import FAISSVectorStore
from skill_retriever.workflows.pipeline import RetrievalPipeline

if TYPE_CHECKING:
    from collections.abc import Generator

# Paths to fixture files
FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALIDATION_PAIRS_PATH = FIXTURES_DIR / "validation_pairs.json"
SEED_DATA_PATH = FIXTURES_DIR / "seed_data.json"

# Embedding dimensions (must match EMBEDDING_CONFIG)
EMBEDDING_DIM = 384


def _type_from_string(type_str: str) -> ComponentType:
    """Convert string type to ComponentType enum."""
    return ComponentType(type_str)


def _edge_type_from_string(type_str: str) -> EdgeType:
    """Convert string edge type to EdgeType enum."""
    return EdgeType(type_str.lower())


@pytest.fixture
def validation_pairs() -> list[dict[str, Any]]:
    """Load validation pairs from JSON fixture.

    Returns:
        List of validation pair dictionaries with query_id, query, expected, category.
    """
    data = json.loads(VALIDATION_PAIRS_PATH.read_text(encoding="utf-8"))
    return data["pairs"]


@pytest.fixture
def validation_qrels(validation_pairs: list[dict[str, Any]]) -> Qrels:
    """Convert validation pairs to ranx Qrels format.

    Qrels format: {query_id: {doc_id: relevance_score, ...}, ...}

    Args:
        validation_pairs: List of validation pair dictionaries.

    Returns:
        ranx Qrels object for MRR evaluation.
    """
    qrels_dict: dict[str, dict[str, int]] = {}
    for pair in validation_pairs:
        query_id = pair["query_id"]
        qrels_dict[query_id] = pair["expected"]
    return Qrels(qrels_dict)


@pytest.fixture
def seed_data() -> dict[str, Any]:
    """Load seed data from JSON fixture.

    Returns:
        Dictionary with 'components', 'edges', and 'metadata' keys.
    """
    return json.loads(SEED_DATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def seeded_pipeline(seed_data: dict[str, Any]) -> Generator[RetrievalPipeline]:
    """Create a RetrievalPipeline with deterministic test data.

    Loads mock components from seed_data fixture into graph and vector stores.
    Uses deterministic RNG (seed=42) for embeddings.

    Args:
        seed_data: Dictionary containing components and edges.

    Yields:
        Configured RetrievalPipeline instance with mock data.
    """
    # Create stores
    graph_store = NetworkXGraphStore()
    vector_store = FAISSVectorStore(dimensions=EMBEDDING_DIM)

    # Deterministic RNG for embeddings
    rng = np.random.default_rng(42)

    # Add nodes and embeddings
    for component in seed_data["components"]:
        comp_id = component["id"]
        comp_type = _type_from_string(component["type"])
        label = component["name"]

        # Add graph node
        node = GraphNode(
            id=comp_id,
            component_type=comp_type,
            label=label,
            embedding_id=comp_id,
        )
        graph_store.add_node(node)

        # Generate deterministic embedding from embedding_text
        # (In real use, this would be from fastembed)
        embedding = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        vector_store.add(comp_id, embedding)

    # Add edges
    for edge in seed_data["edges"]:
        graph_edge = GraphEdge(
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=_edge_type_from_string(edge["type"]),
            weight=1.0,
        )
        graph_store.add_edge(graph_edge)

    # Create pipeline
    pipeline = RetrievalPipeline(
        graph_store=graph_store,
        vector_store=vector_store,
        token_budget=2000,
        cache_size=64,
    )

    yield pipeline

    # Cleanup (cache clear)
    pipeline.clear_cache()


@pytest.fixture
def seed_graph_store(seed_data: dict[str, Any]) -> NetworkXGraphStore:
    """Create a standalone graph store with seed data (for unit tests).

    Args:
        seed_data: Dictionary containing components and edges.

    Returns:
        NetworkXGraphStore populated with seed data.
    """
    graph_store = NetworkXGraphStore()

    for component in seed_data["components"]:
        node = GraphNode(
            id=component["id"],
            component_type=_type_from_string(component["type"]),
            label=component["name"],
            embedding_id=component["id"],
        )
        graph_store.add_node(node)

    for edge in seed_data["edges"]:
        graph_edge = GraphEdge(
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=_edge_type_from_string(edge["type"]),
            weight=1.0,
        )
        graph_store.add_edge(graph_edge)

    return graph_store


@pytest.fixture
def seed_vector_store(seed_data: dict[str, Any]) -> FAISSVectorStore:
    """Create a standalone vector store with seed data (for unit tests).

    Args:
        seed_data: Dictionary containing components.

    Returns:
        FAISSVectorStore populated with deterministic embeddings.
    """
    vector_store = FAISSVectorStore(dimensions=EMBEDDING_DIM)
    rng = np.random.default_rng(42)

    for component in seed_data["components"]:
        comp_id = component["id"]
        embedding = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        vector_store.add(comp_id, embedding)

    return vector_store
