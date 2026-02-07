"""Vector search node with text embedding generation and type filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skill_retriever.config import EMBEDDING_CONFIG
from skill_retriever.nodes.retrieval.models import RankedComponent

if TYPE_CHECKING:
    from fastembed import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]
    from skill_retriever.entities.components import ComponentType
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.memory.vector_store import FAISSVectorStore

# Module-level singleton for expensive TextEmbedding model
_embedding_model: "TextEmbedding | None" = None


def _get_embedding_model() -> "TextEmbedding":
    """Get or create the TextEmbedding model singleton (lazy-loads fastembed)."""
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]  # noqa: F811
        _embedding_model = TextEmbedding(
            model_name=EMBEDDING_CONFIG.model_name,
            cache_dir=EMBEDDING_CONFIG.cache_dir,
        )
    return _embedding_model


def search_by_text(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int = 10,
) -> list[RankedComponent]:
    """Search for components by text query using vector similarity.

    Args:
        query: Natural language search query.
        vector_store: FAISS vector store to search.
        top_k: Maximum number of results to return.

    Returns:
        List of RankedComponent objects sorted by score descending.
    """
    model = _get_embedding_model()

    # Generate embedding for query
    embeddings = list(model.embed([query]))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    query_embedding = embeddings[0]

    # Search vector store
    results = vector_store.search(query_embedding, top_k)  # pyright: ignore[reportUnknownArgumentType]

    # Convert to RankedComponent list
    ranked: list[RankedComponent] = []
    for rank, (component_id, score) in enumerate(results, start=1):
        ranked.append(
            RankedComponent(
                component_id=component_id,
                score=score,
                rank=rank,
                source="vector",
            )
        )

    return ranked


def search_with_type_filter(
    query: str,
    vector_store: FAISSVectorStore,
    graph_store: GraphStore,
    component_type: ComponentType | None = None,
    top_k: int = 10,
) -> list[RankedComponent]:
    """Search for components by text with optional type filtering.

    Type filtering happens AFTER vector retrieval to preserve score ordering.
    Fetches 3x top_k results to allow filtering while maintaining result count.

    Args:
        query: Natural language search query.
        vector_store: FAISS vector store to search.
        graph_store: Graph store for looking up component types.
        component_type: Optional type to filter results. None returns all types.
        top_k: Maximum number of results to return.

    Returns:
        List of RankedComponent objects matching the type filter.
    """
    # Fetch 3x results to allow filtering
    fetch_count = top_k * 3 if component_type is not None else top_k
    candidates = search_by_text(query, vector_store, fetch_count)

    # If no type filter, return first top_k
    if component_type is None:
        return candidates[:top_k]

    # Filter by component type
    filtered: list[RankedComponent] = []
    for candidate in candidates:
        node = graph_store.get_node(candidate.component_id)
        if node is not None and node.component_type == component_type:
            filtered.append(candidate)
            if len(filtered) >= top_k:
                break

    # Re-rank filtered results starting from 1
    return [
        RankedComponent(
            component_id=c.component_id,
            score=c.score,
            rank=new_rank,
            source=c.source,
        )
        for new_rank, c in enumerate(filtered, start=1)
    ]
