"""Query planning and entity extraction for retrieval optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from skill_retriever.nodes.retrieval.models import (
    AbstractionLevel,
    QueryComplexity,
    RetrievalPlan,
)

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

    Abstraction level awareness (RETR-06):
    - SIMPLE queries → LOW abstraction (commands, hooks, settings)
    - MODERATE queries → MEDIUM abstraction (skills)
    - COMPLEX queries → HIGH abstraction (agents, MCPs)
    """
    query_len = len(query)
    query_lower = query.lower()

    # Detect explicit abstraction hints in query
    high_hints = {"agent", "autonomous", "orchestrat", "workflow", "mcp", "server"}
    low_hints = {"command", "hook", "setting", "config", "sandbox", "quick", "simple"}

    has_high_hint = any(hint in query_lower for hint in high_hints)
    has_low_hint = any(hint in query_lower for hint in low_hints)

    # SIMPLE: short query with few entities → suggest low-abstraction components
    if query_len < 300 and entity_count <= 2:
        abstraction = AbstractionLevel.LOW
        suggested = ["command", "hook", "setting", "sandbox"]
        if has_high_hint:
            abstraction = AbstractionLevel.HIGH
            suggested = ["agent", "mcp"]

        return RetrievalPlan(
            complexity=QueryComplexity.SIMPLE,
            use_ppr=False,
            use_flow_pruning=False,
            ppr_alpha=0.85,
            max_results=10,
            abstraction_level=abstraction,
            suggested_types=suggested,
        )

    # COMPLEX: long query or many entities → suggest high-abstraction components
    if query_len > 600 or entity_count > 5:
        abstraction = AbstractionLevel.HIGH
        suggested = ["agent", "mcp", "skill"]
        if has_low_hint:
            abstraction = AbstractionLevel.LOW
            suggested = ["command", "hook", "setting"]

        return RetrievalPlan(
            complexity=QueryComplexity.COMPLEX,
            use_ppr=True,
            use_flow_pruning=True,
            ppr_alpha=0.7,
            max_results=30,
            abstraction_level=abstraction,
            suggested_types=suggested,
        )

    # MODERATE: everything else → suggest medium-abstraction components
    abstraction = AbstractionLevel.MEDIUM
    suggested = ["skill", "command", "agent"]
    if has_high_hint:
        abstraction = AbstractionLevel.HIGH
        suggested = ["agent", "mcp", "skill"]
    elif has_low_hint:
        abstraction = AbstractionLevel.LOW
        suggested = ["command", "hook", "setting"]

    return RetrievalPlan(
        complexity=QueryComplexity.MODERATE,
        use_ppr=True,
        use_flow_pruning=False,
        ppr_alpha=0.85,
        max_results=20,
        abstraction_level=abstraction,
        suggested_types=suggested,
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
