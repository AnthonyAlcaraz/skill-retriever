"""Tests for the two-phase entity resolution pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from skill_retriever.entities import ComponentMetadata, ComponentType
from skill_retriever.nodes.ingestion.resolver import EntityResolver


def _make_entity(
    name: str,
    component_type: ComponentType = ComponentType.AGENT,
    description: str = "",
    tags: list[str] | None = None,
    tools: list[str] | None = None,
    last_updated: datetime | None = None,
    owner: str = "test",
    repo: str = "repo",
) -> ComponentMetadata:
    """Helper to create a ComponentMetadata for testing."""
    return ComponentMetadata(
        id=ComponentMetadata.generate_id(owner, repo, component_type, name),
        name=name,
        component_type=component_type,
        description=description,
        tags=tags or [],
        tools=tools or [],
        last_updated=last_updated,
    )


class TestEntityResolver:
    """Tests for EntityResolver."""

    def test_resolve_exact_duplicates(self) -> None:
        """Two agents with the same name should be merged into one."""
        resolver = EntityResolver(fuzzy_threshold=80.0)
        entities = [
            _make_entity("code-reviewer", description="Reviews code"),
            _make_entity("code-reviewer", description="Reviews code changes"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        assert result[0].name == "code-reviewer"

    def test_resolve_similar_names(self) -> None:
        """Entities with similar names (hyphen vs underscore) should merge."""
        resolver = EntityResolver(fuzzy_threshold=80.0)
        entities = [
            _make_entity("code-reviewer", description="Reviews code"),
            _make_entity("code_reviewer", description="Reviews code too"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1

    def test_resolve_different_types_not_merged(self) -> None:
        """Same name but different component_type should NOT merge (blocking)."""
        resolver = EntityResolver(fuzzy_threshold=80.0)
        entities = [
            _make_entity(
                "code-reviewer",
                component_type=ComponentType.AGENT,
                description="Agent reviewer",
            ),
            _make_entity(
                "code-reviewer",
                component_type=ComponentType.SKILL,
                description="Skill reviewer",
            ),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 2

    def test_resolve_different_descriptions_not_merged_with_embeddings(self) -> None:
        """Similar names but very different descriptions should NOT merge
        when an embedding model is provided and descriptions diverge enough.

        In fuzzy-only mode they WILL merge since names match.
        """
        # Fuzzy-only mode: names match, so they merge
        resolver_fuzzy = EntityResolver(fuzzy_threshold=80.0)
        entities = [
            _make_entity(
                "data-processor",
                description="Handles CSV file parsing and transformation",
            ),
            _make_entity(
                "data-processor",
                description="Manages PostgreSQL database connections and queries",
            ),
        ]
        result_fuzzy = resolver_fuzzy.resolve(entities)
        assert len(result_fuzzy) == 1, "Fuzzy-only mode should merge same-name entities"

        # With a mock embedding model that returns very different embeddings
        class _MockEmbedding:
            """Returns orthogonal embeddings to simulate different descriptions."""

            _call_count: int = 0

            def embed(self, documents: list[str]) -> list[list[float]]:
                results = []
                for i, _doc in enumerate(documents):
                    vec = [0.0] * 10
                    vec[(self._call_count + i) % 10] = 1.0
                    results.append(vec)
                self._call_count += len(documents)
                return results

        resolver_emb = EntityResolver(
            embedding_model=_MockEmbedding(),  # type: ignore[arg-type]
            fuzzy_threshold=80.0,
            embedding_threshold=0.85,
        )
        result_emb = resolver_emb.resolve(entities)
        assert len(result_emb) == 2, (
            "Embedding confirmation should reject divergent descriptions"
        )

    def test_resolve_empty_input(self) -> None:
        """Empty input should return empty output."""
        resolver = EntityResolver(fuzzy_threshold=80.0)
        result = resolver.resolve([])
        assert result == []

    def test_resolve_no_duplicates(self) -> None:
        """All unique entities should be returned unchanged."""
        resolver = EntityResolver(fuzzy_threshold=80.0)
        entities = [
            _make_entity("alpha-agent", description="Does alpha things"),
            _make_entity("beta-agent", description="Does beta things"),
            _make_entity("gamma-agent", description="Does gamma things"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 3

    def test_merge_preserves_richest_metadata(self) -> None:
        """Merged entity should have union of tags/tools and longest description."""
        resolver = EntityResolver(fuzzy_threshold=80.0)

        earlier = datetime(2024, 1, 1, tzinfo=timezone.utc)
        later = datetime(2024, 6, 15, tzinfo=timezone.utc)

        entities = [
            _make_entity(
                "code-reviewer",
                description="Short desc",
                tags=["python", "review"],
                tools=["lint"],
                last_updated=earlier,
            ),
            _make_entity(
                "code-reviewer",
                description="A much longer and more detailed description of the code reviewer component",
                tags=["review", "quality"],
                tools=["lint", "format"],
                last_updated=later,
            ),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
        merged = result[0]

        # Should keep the longest description
        assert "much longer" in merged.description

        # Tags should be union
        assert set(merged.tags) == {"python", "review", "quality"}

        # Tools should be union
        assert set(merged.tools) == {"lint", "format"}

        # Should keep the most recent last_updated
        assert merged.last_updated == later

    def test_resolve_transitive_duplicates(self) -> None:
        """If A~B and B~C, all three should merge into one group."""
        resolver = EntityResolver(fuzzy_threshold=70.0)
        entities = [
            _make_entity("code reviewer", description="Reviews code"),
            _make_entity("code-reviewer", description="Reviews code"),
            _make_entity("code_reviewer", description="Reviews code"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 1
