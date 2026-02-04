"""Query planning and entity extraction for retrieval optimization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz, process

from skill_retriever.nodes.retrieval.models import (
    AbstractionLevel,
    QueryComplexity,
    RetrievalPlan,
)

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore

logger = logging.getLogger(__name__)

STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "to", "for", "with", "how", "what",
    "do", "does", "can", "could", "should", "would", "i", "me", "my",
    "we", "our", "you", "your", "it", "its", "this", "that", "these",
    "those", "and", "or", "but", "in", "on", "at", "by", "of", "from",
})

# Fuzzy matching thresholds (RETR-07)
FUZZY_EXACT_THRESHOLD = 95  # Treat as exact match
FUZZY_HIGH_THRESHOLD = 85   # Strong match
FUZZY_MIN_THRESHOLD = 70    # Minimum for inclusion

# Synonyms for common terms (RETR-07)
SYNONYMS: dict[str, set[str]] = {
    "git": {"version-control", "vcs", "repository", "repo"},
    "commit": {"checkin", "save", "push"},
    "test": {"testing", "spec", "unittest", "pytest"},
    "deploy": {"deployment", "release", "ship", "publish"},
    "auth": {"authentication", "login", "signin", "oauth"},
    "api": {"endpoint", "rest", "graphql", "http"},
    "db": {"database", "sql", "postgres", "mysql", "sqlite"},
    "aws": {"amazon", "cloud", "s3", "lambda", "ec2"},
    "ci": {"continuous-integration", "pipeline", "github-actions", "jenkins"},
    "cd": {"continuous-deployment", "delivery"},
    "docker": {"container", "containerize", "k8s", "kubernetes"},
    "lint": {"linting", "eslint", "ruff", "format", "formatter"},
}


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


def _expand_with_synonyms(tokens: list[str]) -> set[str]:
    """Expand tokens with known synonyms (RETR-07).

    Args:
        tokens: List of query tokens.

    Returns:
        Set of tokens including synonyms.
    """
    expanded = set(tokens)
    for token in tokens:
        # Check if token is a key in synonyms
        if token in SYNONYMS:
            expanded.update(SYNONYMS[token])
        # Check if token is a value in synonyms (reverse lookup)
        for key, values in SYNONYMS.items():
            if token in values:
                expanded.add(key)
                expanded.update(values)
    return expanded


def _build_label_index(graph_store: GraphStore) -> dict[str, list[str]]:
    """Build inverted index from labels to node IDs for fast lookup.

    Args:
        graph_store: Graph store to index.

    Returns:
        Dict mapping lowercase labels to list of node IDs.
    """
    from skill_retriever.memory.graph_store import NetworkXGraphStore

    label_to_ids: dict[str, list[str]] = {}

    if isinstance(graph_store, NetworkXGraphStore):
        nx_graph: Any = graph_store._graph  # pyright: ignore[reportPrivateUsage]
        for node_id in nx_graph.nodes:
            node_data: dict[str, Any] = dict(nx_graph.nodes[node_id])
            label = str(node_data.get("label", "")).lower()
            if label:
                if label not in label_to_ids:
                    label_to_ids[label] = []
                label_to_ids[label].append(str(node_id))

    return label_to_ids


def extract_query_entities(query: str, graph_store: GraphStore) -> set[str]:
    """Extract entity IDs from query using fuzzy matching against graph node labels (RETR-07).

    Tokenizes by whitespace, strips punctuation, filters stopwords,
    expands with synonyms, then fuzzy-matches against graph node labels.

    Uses RapidFuzz for efficient fuzzy matching with configurable thresholds.

    Returns:
        Set of node IDs that match query tokens (exact, fuzzy, or synonym).
    """
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

    # Expand with synonyms
    expanded = _expand_with_synonyms(filtered)

    # Build label index for fuzzy matching
    label_index = _build_label_index(graph_store)
    if not label_index:
        return set()

    all_labels = list(label_index.keys())
    matched_ids: set[str] = set()

    for token in expanded:
        # Phase 1: Exact substring match (fast path)
        for label, node_ids in label_index.items():
            if token in label or label in token:
                matched_ids.update(node_ids)

        # Phase 2: Fuzzy match if no exact matches found for this token
        if not any(token in label or label in token for label in all_labels):
            # Use RapidFuzz to find best matches
            matches = process.extract(
                token,
                all_labels,
                scorer=fuzz.token_sort_ratio,
                limit=5,
                score_cutoff=FUZZY_MIN_THRESHOLD,
            )

            for match_label, score, _ in matches:
                if score >= FUZZY_MIN_THRESHOLD:
                    matched_ids.update(label_index[match_label])
                    if score >= FUZZY_HIGH_THRESHOLD:
                        logger.debug(
                            "Fuzzy match: '%s' -> '%s' (score: %d)",
                            token, match_label, score
                        )

    return matched_ids
