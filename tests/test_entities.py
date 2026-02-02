"""Tests for entity models: ComponentType, ComponentMetadata, GraphNode, GraphEdge, EdgeType."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from skill_retriever.entities import (
    ComponentMetadata,
    ComponentType,
    EdgeType,
    GraphEdge,
    GraphNode,
)


class TestComponentType:
    def test_component_type_values(self) -> None:
        expected = {"agent", "skill", "command", "setting", "mcp", "hook", "sandbox"}
        actual = {ct.value for ct in ComponentType}
        assert actual == expected
        assert len(ComponentType) == 7
        for ct in ComponentType:
            assert isinstance(ct, str)
            assert ct == ct.lower()


class TestComponentMetadata:
    def test_component_metadata_creation(self) -> None:
        now = datetime.now(tz=UTC)
        meta = ComponentMetadata(
            id="owner/repo/agent/my-agent",
            name="My Agent",
            component_type=ComponentType.AGENT,
            description="A test agent",
            tags=["test", "demo"],
            author="testauthor",
            version="1.0.0",
            last_updated=now,
            commit_count=42,
            commit_frequency_30d=1.4,
            raw_content="# My Agent\nDoes things.",
            parameters={"model": "opus"},
            dependencies=["owner/repo/skill/helper"],
            tools=["Bash", "Read"],
            source_repo="owner/repo",
            source_path="agents/my-agent.md",
            category="ai-specialists",
        )
        dumped = meta.model_dump()
        restored = ComponentMetadata.model_validate(dumped)
        assert restored == meta
        assert restored.id == "owner/repo/agent/my-agent"
        assert restored.commit_count == 42

    def test_component_metadata_defaults(self) -> None:
        meta = ComponentMetadata(
            id="owner/repo/skill/basic",
            name="Basic",
            component_type=ComponentType.SKILL,
        )
        assert meta.description == ""
        assert meta.tags == []
        assert meta.author == ""
        assert meta.version == ""
        assert meta.last_updated is None
        assert meta.commit_count == 0
        assert meta.commit_frequency_30d == 0.0
        assert meta.raw_content == ""
        assert meta.parameters == {}
        assert meta.dependencies == []
        assert meta.tools == []
        assert meta.source_repo == ""
        assert meta.source_path == ""
        assert meta.category == ""

    def test_generate_id(self) -> None:
        result = ComponentMetadata.generate_id(
            "davila7", "claude-code-templates", ComponentType.AGENT, "Prompt Engineer"
        )
        assert result == "davila7/claude-code-templates/agent/prompt-engineer"

    def test_generate_id_special_chars(self) -> None:
        assert (
            ComponentMetadata.generate_id("o", "r", ComponentType.HOOK, "  My Cool Hook  ")
            == "o/r/hook/my-cool-hook"
        )
        assert (
            ComponentMetadata.generate_id("o", "r", ComponentType.MCP, "UPPER Case NAME")
            == "o/r/mcp/upper-case-name"
        )
        assert (
            ComponentMetadata.generate_id("o", "r", ComponentType.COMMAND, "multi   spaces")
            == "o/r/command/multi-spaces"
        )

    def test_component_metadata_frozen(self) -> None:
        meta = ComponentMetadata(
            id="owner/repo/agent/frozen",
            name="Frozen",
            component_type=ComponentType.AGENT,
        )
        with pytest.raises(ValidationError):
            meta.name = "Changed"  # type: ignore[misc]

    def test_component_metadata_model_copy(self) -> None:
        original = ComponentMetadata(
            id="owner/repo/agent/copy-test",
            name="CopyTest",
            component_type=ComponentType.AGENT,
            description="original",
        )
        updated = original.model_copy(update={"description": "new"})
        assert updated.description == "new"
        assert original.description == "original"
        assert updated.id == original.id

    def test_id_validator_normalizes(self) -> None:
        meta = ComponentMetadata(
            id="  Owner/Repo/agent/My Agent Name  ",
            name="X",
            component_type=ComponentType.AGENT,
        )
        assert meta.id == "Owner/Repo/agent/my-agent-name"


class TestGraphNode:
    def test_graph_node_creation(self) -> None:
        node = GraphNode(
            id="owner/repo/skill/helper",
            component_type=ComponentType.SKILL,
            label="Helper Skill",
            embedding_id="emb-123",
        )
        assert node.id == "owner/repo/skill/helper"
        assert node.component_type == ComponentType.SKILL
        assert node.label == "Helper Skill"
        assert node.embedding_id == "emb-123"


class TestGraphEdge:
    def test_graph_edge_creation(self) -> None:
        edge = GraphEdge(
            source_id="a/b/agent/x",
            target_id="a/b/skill/y",
            edge_type=EdgeType.DEPENDS_ON,
            weight=0.8,
            metadata={"reason": "uses tool"},
        )
        assert edge.source_id == "a/b/agent/x"
        assert edge.target_id == "a/b/skill/y"
        assert edge.edge_type == EdgeType.DEPENDS_ON
        assert edge.weight == 0.8
        assert edge.metadata == {"reason": "uses tool"}

    def test_edge_type_values(self) -> None:
        expected = {"depends_on", "enhances", "conflicts_with", "bundles_with", "same_category"}
        actual = {et.value for et in EdgeType}
        assert actual == expected
        assert len(EdgeType) == 5
