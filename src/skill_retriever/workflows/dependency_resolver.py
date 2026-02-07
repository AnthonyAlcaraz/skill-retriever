"""Dependency resolution and conflict detection for component retrieval."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

from skill_retriever.entities.graph import EdgeType
from skill_retriever.workflows.models import ConflictInfo

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore

logger = logging.getLogger(__name__)


def resolve_transitive_dependencies(
    component_ids: list[str],
    graph_store: GraphStore,
) -> tuple[set[str], list[str]]:
    """Resolve all transitive dependencies for the given components.

    Uses nx.descendants() on a subgraph containing only DEPENDS_ON edges
    to find all transitive dependencies.

    Args:
        component_ids: List of component IDs to resolve dependencies for.
        graph_store: Graph store containing component relationships.

    Returns:
        Tuple of (all_component_ids, newly_added_dependency_ids)
        - all_component_ids: Original components plus all their transitive deps
        - newly_added_dependency_ids: Only the deps that weren't in original list
    """
    if not component_ids:
        return set(), []

    graph = graph_store.nx_graph
    depends_on_subgraph = graph_store.get_depends_on_subgraph()

    # Check for cycles (log warning but continue)
    if not nx.is_directed_acyclic_graph(depends_on_subgraph):
        logger.warning(
            "Dependency graph contains cycles - transitive resolution may be incomplete"
        )

    original_set = set(component_ids)
    all_deps: set[str] = set()

    for component_id in component_ids:
        # Skip components not in graph
        if component_id not in graph:
            logger.debug("Component %s not found in graph, skipping", component_id)
            continue

        # Get transitive dependencies via descendants in the DEPENDS_ON subgraph
        if component_id in depends_on_subgraph:
            transitive: set[str] = nx.descendants(  # pyright: ignore[reportUnknownMemberType]
                depends_on_subgraph, component_id
            )
            all_deps.update(transitive)

    # Add original components to result set
    all_component_ids = original_set | all_deps

    # Calculate which deps were newly added (not in original list)
    newly_added = sorted(all_deps - original_set)

    return all_component_ids, newly_added


def detect_conflicts(
    component_ids: set[str],
    graph_store: GraphStore,
) -> list[ConflictInfo]:
    """Find all CONFLICTS_WITH relationships among the given components.

    Checks both directions (A conflicts B and B conflicts A are the same conflict).

    Args:
        component_ids: Set of component IDs to check for conflicts.
        graph_store: Graph store containing component relationships.

    Returns:
        List of ConflictInfo for each detected conflict pair.
    """
    if not component_ids:
        return []

    # Track checked pairs to avoid duplicates (A,B same as B,A)
    checked: set[frozenset[str]] = set()
    conflicts: list[ConflictInfo] = []

    for component_id in component_ids:
        edges = graph_store.get_edges(component_id)

        for edge in edges:
            if edge.edge_type != EdgeType.CONFLICTS_WITH:
                continue

            # Determine the "other" component in the conflict
            other_id = (
                edge.target_id if edge.source_id == component_id else edge.source_id
            )

            # Only record if other component is in our selection set
            if other_id not in component_ids:
                continue

            # Create unique pair key to avoid duplicates
            pair = frozenset({component_id, other_id})
            if pair in checked:
                continue
            checked.add(pair)

            # Extract reason from edge metadata
            reason = edge.metadata.get("reason", "Component conflict detected")

            # Order components lexicographically for deterministic output
            comp_a, comp_b = sorted([component_id, other_id])
            conflicts.append(
                ConflictInfo(
                    component_a=comp_a,
                    component_b=comp_b,
                    reason=str(reason),
                )
            )

    return sorted(conflicts, key=lambda c: (c.component_a, c.component_b))
