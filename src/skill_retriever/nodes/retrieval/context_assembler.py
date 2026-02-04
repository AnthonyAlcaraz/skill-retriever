"""Token-budgeted context assembly with type priority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skill_retriever.entities.components import ComponentType

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.nodes.retrieval.models import RankedComponent

# Approximate token estimation: 4 chars = 1 token (conservative)
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 2000

# Priority ordering for component types (lower number = higher priority)
TYPE_PRIORITY: dict[ComponentType, int] = {
    ComponentType.AGENT: 1,
    ComponentType.SKILL: 2,
    ComponentType.COMMAND: 3,
    ComponentType.MCP: 4,
    ComponentType.HOOK: 5,
    ComponentType.SETTING: 6,
    ComponentType.SANDBOX: 7,
}


@dataclass
class RetrievalContext:
    """Result of context assembly with token budgeting."""

    components: list[RankedComponent]
    total_tokens: int
    truncated: bool
    excluded_count: int


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def get_component_content(component_id: str, graph_store: GraphStore) -> str:
    """Get component content for token estimation.

    Currently returns node label. Future phases will pull full raw_content.
    """
    node = graph_store.get_node(component_id)
    if node is None:
        return ""
    return node.label


def assemble_context(
    ranked_components: list[RankedComponent],
    graph_store: GraphStore,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> RetrievalContext:
    """Assemble context from ranked components within token budget.

    Components are sorted by:
    1. Type priority (agents before skills before commands)
    2. Score descending within same type
    """
    if not ranked_components:
        return RetrievalContext(
            components=[],
            total_tokens=0,
            truncated=False,
            excluded_count=0,
        )

    # Filter out stub nodes (auto-created by edges, have no real content)
    valid_components = [
        comp
        for comp in ranked_components
        if graph_store.get_node(comp.component_id) is not None
    ]

    # Get component types for sorting
    def get_sort_key(comp: RankedComponent) -> tuple[int, float]:
        node = graph_store.get_node(comp.component_id)
        # Node guaranteed to exist after filtering, but satisfy type checker
        type_priority = (
            999 if node is None else TYPE_PRIORITY.get(node.component_type, 999)
        )
        # Negate score for descending order within same type
        return (type_priority, -comp.score)

    sorted_components = sorted(valid_components, key=get_sort_key)

    included: list[RankedComponent] = []
    total_tokens = 0
    truncated = False

    for comp in sorted_components:
        content = get_component_content(comp.component_id, graph_store)
        tokens = estimate_tokens(content)

        if total_tokens + tokens > token_budget:
            truncated = True
            break

        included.append(comp)
        total_tokens += tokens

    excluded_count = len(ranked_components) - len(included)

    return RetrievalContext(
        components=included,
        total_tokens=total_tokens,
        truncated=truncated,
        excluded_count=excluded_count,
    )
