"""Tests for dependency resolution and conflict detection."""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.workflows.dependency_resolver import (
    detect_conflicts,
    resolve_transitive_dependencies,
)


@pytest.fixture
def graph_store() -> NetworkXGraphStore:
    """Graph store with test components and dependency/conflict edges."""
    store = NetworkXGraphStore()

    # Create components: A -> B -> C (dependency chain)
    # D conflicts with B
    store.add_node(
        GraphNode(
            id="comp-a",
            label="Component A",
            component_type=ComponentType.AGENT,
            embedding_id="emb-a",
        )
    )
    store.add_node(
        GraphNode(
            id="comp-b",
            label="Component B",
            component_type=ComponentType.SKILL,
            embedding_id="emb-b",
        )
    )
    store.add_node(
        GraphNode(
            id="comp-c",
            label="Component C",
            component_type=ComponentType.SKILL,
            embedding_id="emb-c",
        )
    )
    store.add_node(
        GraphNode(
            id="comp-d",
            label="Component D",
            component_type=ComponentType.COMMAND,
            embedding_id="emb-d",
        )
    )
    store.add_node(
        GraphNode(
            id="comp-e",
            label="Component E (no deps)",
            component_type=ComponentType.SKILL,
            embedding_id="emb-e",
        )
    )

    # Dependency chain: A depends on B, B depends on C
    store.add_edge(
        GraphEdge(
            source_id="comp-a",
            target_id="comp-b",
            edge_type=EdgeType.DEPENDS_ON,
        )
    )
    store.add_edge(
        GraphEdge(
            source_id="comp-b",
            target_id="comp-c",
            edge_type=EdgeType.DEPENDS_ON,
        )
    )

    # Conflict: D conflicts with B
    store.add_edge(
        GraphEdge(
            source_id="comp-d",
            target_id="comp-b",
            edge_type=EdgeType.CONFLICTS_WITH,
            metadata={"reason": "Incompatible authentication methods"},
        )
    )

    # Enhances edge (should NOT be followed for dependencies)
    store.add_edge(
        GraphEdge(
            source_id="comp-e",
            target_id="comp-a",
            edge_type=EdgeType.ENHANCES,
        )
    )

    return store


class TestResolveTransitiveDependencies:
    """Tests for resolve_transitive_dependencies function."""

    def test_resolve_empty_list(self, graph_store: NetworkXGraphStore) -> None:
        """Empty input returns empty output."""
        all_ids, newly_added = resolve_transitive_dependencies([], graph_store)
        assert all_ids == set()
        assert newly_added == []

    def test_resolve_no_deps(self, graph_store: NetworkXGraphStore) -> None:
        """Component with no DEPENDS_ON edges returns only itself."""
        all_ids, newly_added = resolve_transitive_dependencies(
            ["comp-e"], graph_store
        )
        assert all_ids == {"comp-e"}
        assert newly_added == []

    def test_resolve_single_hop(self, graph_store: NetworkXGraphStore) -> None:
        """A depends on B, returns {A, B}."""
        # B depends on C, but we're only asking about B
        all_ids, newly_added = resolve_transitive_dependencies(
            ["comp-b"], graph_store
        )
        # B's only dependency is C
        assert all_ids == {"comp-b", "comp-c"}
        assert newly_added == ["comp-c"]

    def test_resolve_transitive(self, graph_store: NetworkXGraphStore) -> None:
        """A depends on B, B depends on C, returns {A, B, C}."""
        all_ids, newly_added = resolve_transitive_dependencies(
            ["comp-a"], graph_store
        )
        assert all_ids == {"comp-a", "comp-b", "comp-c"}
        # B and C were added as dependencies
        assert set(newly_added) == {"comp-b", "comp-c"}

    def test_resolve_missing_node(self, graph_store: NetworkXGraphStore) -> None:
        """Component not in graph is handled gracefully."""
        all_ids, newly_added = resolve_transitive_dependencies(
            ["nonexistent"], graph_store
        )
        # Missing component still in original set, no deps added
        assert all_ids == {"nonexistent"}
        assert newly_added == []

    def test_resolve_mixed_valid_invalid(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Mix of valid and invalid components."""
        all_ids, _newly_added = resolve_transitive_dependencies(
            ["comp-a", "nonexistent"], graph_store
        )
        # comp-a's deps (B, C) added, nonexistent passed through
        assert "comp-a" in all_ids
        assert "nonexistent" in all_ids
        assert "comp-b" in all_ids
        assert "comp-c" in all_ids

    def test_resolve_ignores_enhances_edges(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """ENHANCES edges are not followed for dependency resolution."""
        # E enhances A, but that's not a dependency
        all_ids, newly_added = resolve_transitive_dependencies(
            ["comp-e"], graph_store
        )
        # E has no DEPENDS_ON edges, so only E returned
        assert all_ids == {"comp-e"}
        assert newly_added == []

    def test_resolve_deduplicates_common_deps(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Multiple components sharing deps don't duplicate."""
        all_ids, newly_added = resolve_transitive_dependencies(
            ["comp-a", "comp-b"], graph_store
        )
        # A depends on B, C. B depends on C. Result is A, B, C (C not duplicated)
        assert all_ids == {"comp-a", "comp-b", "comp-c"}
        # Only C was newly added (B was in original list)
        assert newly_added == ["comp-c"]


class TestDetectConflicts:
    """Tests for detect_conflicts function."""

    def test_detect_no_conflicts(self, graph_store: NetworkXGraphStore) -> None:
        """Components with no conflicts returns empty list."""
        conflicts = detect_conflicts({"comp-a", "comp-c"}, graph_store)
        assert conflicts == []

    def test_detect_single_conflict(self, graph_store: NetworkXGraphStore) -> None:
        """A conflicts with B, both in set, returns ConflictInfo."""
        # D conflicts with B
        conflicts = detect_conflicts({"comp-d", "comp-b"}, graph_store)
        assert len(conflicts) == 1
        assert conflicts[0].component_a == "comp-b"  # Lexicographically first
        assert conflicts[0].component_b == "comp-d"
        assert conflicts[0].reason == "Incompatible authentication methods"

    def test_detect_bidirectional_check(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Conflict stored A->B still found when checking from B's edges."""
        # The edge is D -> B, but checking B's edges should also find it
        conflicts = detect_conflicts({"comp-b", "comp-d"}, graph_store)
        assert len(conflicts) == 1
        assert {conflicts[0].component_a, conflicts[0].component_b} == {
            "comp-b",
            "comp-d",
        }

    def test_detect_excludes_non_selected(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """A conflicts with B, but B not in selected set, no conflict."""
        # D conflicts with B, but if B is not in the set, no conflict reported
        conflicts = detect_conflicts({"comp-d", "comp-a"}, graph_store)
        assert conflicts == []

    def test_detect_empty_set(self, graph_store: NetworkXGraphStore) -> None:
        """Empty component set returns no conflicts."""
        conflicts = detect_conflicts(set(), graph_store)
        assert conflicts == []

    def test_detect_single_component(self, graph_store: NetworkXGraphStore) -> None:
        """Single component cannot conflict with itself."""
        conflicts = detect_conflicts({"comp-d"}, graph_store)
        assert conflicts == []

    def test_detect_default_reason(self) -> None:
        """Conflict without reason metadata uses default message."""
        store = NetworkXGraphStore()
        store.add_node(
            GraphNode(
                id="x",
                label="X",
                component_type=ComponentType.SKILL,
                embedding_id="emb-x",
            )
        )
        store.add_node(
            GraphNode(
                id="y",
                label="Y",
                component_type=ComponentType.SKILL,
                embedding_id="emb-y",
            )
        )
        store.add_edge(
            GraphEdge(
                source_id="x",
                target_id="y",
                edge_type=EdgeType.CONFLICTS_WITH,
                # No reason in metadata
            )
        )

        conflicts = detect_conflicts({"x", "y"}, store)
        assert len(conflicts) == 1
        assert conflicts[0].reason == "Component conflict detected"


class TestCycleHandling:
    """Tests for cycle detection in dependency graphs."""

    def test_handles_cycle_gracefully(self) -> None:
        """Cycles in dependency graph don't cause infinite loops."""
        store = NetworkXGraphStore()
        # Create cycle: A -> B -> A
        store.add_node(
            GraphNode(
                id="cycle-a",
                label="Cycle A",
                component_type=ComponentType.SKILL,
                embedding_id="emb-cycle-a",
            )
        )
        store.add_node(
            GraphNode(
                id="cycle-b",
                label="Cycle B",
                component_type=ComponentType.SKILL,
                embedding_id="emb-cycle-b",
            )
        )
        store.add_edge(
            GraphEdge(
                source_id="cycle-a",
                target_id="cycle-b",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        store.add_edge(
            GraphEdge(
                source_id="cycle-b",
                target_id="cycle-a",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        # Should complete without hanging (nx.descendants handles cycles)
        all_ids, _newly_added = resolve_transitive_dependencies(
            ["cycle-a"], store
        )
        # Both nodes should be in result
        assert "cycle-a" in all_ids
        assert "cycle-b" in all_ids
