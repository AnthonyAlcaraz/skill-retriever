"""Query planning and entity extraction for retrieval optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skill_retriever.nodes.retrieval.models import QueryComplexity, RetrievalPlan

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore


STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "to", "for", "with", "how", "what",
    "do", "does", "can", "could", "should", "would", "i", "me", "my",
    "we", "our", "you", "your", "it", "its", "this", "that", "these",
    "those", "and", "or", "but", "in", "on", "at", "by", "of", "from",
})


def plan_retrieval(query: str, entity_count: int) -> RetrievalPlan:
    """Determine retrieval strategy based on query length and entity count.

    Classification rules:
    - SIMPLE: query < 300 chars AND entity_count <= 2
    - COMPLEX: query > 600 chars OR entity_count > 5
    - MODERATE: everything else
    """
    query_len = len(query)

    # SIMPLE: short query with few entities
    if query_len < 300 and entity_count <= 2:
        return RetrievalPlan(
            complexity=QueryComplexity.SIMPLE,
            use_ppr=False,
            use_flow_pruning=False,
            ppr_alpha=0.85,
            max_results=10,
        )

    # COMPLEX: long query or many entities
    if query_len > 600 or entity_count > 5:
        return RetrievalPlan(
            complexity=QueryComplexity.COMPLEX,
            use_ppr=True,
            use_flow_pruning=True,
            ppr_alpha=0.7,
            max_results=30,
        )

    # MODERATE: everything else
    return RetrievalPlan(
        complexity=QueryComplexity.MODERATE,
        use_ppr=True,
        use_flow_pruning=False,
        ppr_alpha=0.85,
        max_results=20,
    )


def extract_query_entities(query: str, graph_store: GraphStore) -> set[str]:
    """Extract entity IDs from query by matching tokens against graph node labels.

    Tokenizes by whitespace, strips punctuation, filters stopwords,
    then matches remaining tokens against graph node labels (case-insensitive).

    Returns:
        Set of node IDs that match query tokens.
    """
    from skill_retriever.memory.graph_store import NetworkXGraphStore

    # Tokenize: split on whitespace, strip punctuation
    tokens: list[str] = []
    for word in query.split():
        # Strip leading/trailing punctuation
        clean = word.strip(".,!?;:'\"()[]{}<>-_")
        if clean:
            tokens.append(clean.lower())

    # Filter stopwords
    filtered = [t for t in tokens if t not in STOPWORDS]

    if not filtered:
        return set()

    # Match against graph nodes
    matched_ids: set[str] = set()

    # Use isinstance for proper type narrowing
    if isinstance(graph_store, NetworkXGraphStore):
        # Access NetworkX graph directly for node iteration
        nx_graph: Any = graph_store._graph  # pyright: ignore[reportPrivateUsage]
        for node_id in nx_graph.nodes:
            node_data: dict[str, Any] = dict(nx_graph.nodes[node_id])
            label = str(node_data.get("label", "")).lower()
            for token in filtered:
                if token in label:
                    matched_ids.add(str(node_id))
                    break

    return matched_ids
