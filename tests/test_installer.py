"""Tests for component installer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from skill_retriever.entities.components import ComponentMetadata, ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.mcp.installer import (
    INSTALL_PATHS,
    ComponentInstaller,
    deep_merge,
    install_component,
    merge_settings,
)
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.memory.metadata_store import MetadataStore

if TYPE_CHECKING:
    from pathlib import Path


class TestInstallPaths:
    """Test INSTALL_PATHS constant."""

    def test_all_component_types_mapped(self) -> None:
        """All ComponentType values should have a path mapping."""
        for comp_type in ComponentType:
            assert comp_type in INSTALL_PATHS, f"Missing path for {comp_type}"

    def test_skill_path_format(self) -> None:
        """Skill path should include {name} placeholder in directory."""
        assert "{name}" in INSTALL_PATHS[ComponentType.SKILL]
        assert INSTALL_PATHS[ComponentType.SKILL].endswith("SKILL.md")

    def test_command_path_format(self) -> None:
        """Command path should use {name}.md format."""
        assert "{name}" in INSTALL_PATHS[ComponentType.COMMAND]
        assert INSTALL_PATHS[ComponentType.COMMAND].endswith(".md")

    def test_setting_path_fixed(self) -> None:
        """Setting path should be fixed (no {name} placeholder)."""
        assert "{name}" not in INSTALL_PATHS[ComponentType.SETTING]
        assert INSTALL_PATHS[ComponentType.SETTING] == ".claude/settings.json"


class TestDeepMerge:
    """Test deep_merge function."""

    def test_simple_merge(self) -> None:
        """Merge flat dictionaries."""
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Recursively merge nested dictionaries."""
        base = {"outer": {"a": 1, "b": 2}}
        overlay = {"outer": {"b": 3, "c": 4}}
        result = deep_merge(base, overlay)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_list_extend_dedupe(self) -> None:
        """Lists should be extended and deduped."""
        base = {"items": [1, 2, 3]}
        overlay = {"items": [3, 4, 5]}
        result = deep_merge(base, overlay)
        assert result == {"items": [1, 2, 3, 4, 5]}

    def test_list_preserves_order(self) -> None:
        """List merge should preserve original order."""
        base = {"items": ["a", "b"]}
        overlay = {"items": ["c", "a"]}  # 'a' already exists
        result = deep_merge(base, overlay)
        assert result == {"items": ["a", "b", "c"]}

    def test_overlay_wins_mismatched_types(self) -> None:
        """Overlay value wins when types don't match."""
        base = {"key": [1, 2, 3]}
        overlay = {"key": "not a list"}
        result = deep_merge(base, overlay)
        assert result == {"key": "not a list"}

    def test_does_not_mutate_base(self) -> None:
        """Original base dict should not be mutated."""
        base = {"a": 1, "nested": {"b": 2}}
        overlay = {"nested": {"c": 3}}
        deep_merge(base, overlay)
        assert base == {"a": 1, "nested": {"b": 2}}


class TestMergeSettings:
    """Test merge_settings function."""

    def test_merge_new_file(self, tmp_path: Path) -> None:
        """Merge into non-existent settings creates from overlay."""
        settings_path = tmp_path / "settings.json"
        new_settings = {"model": "opus", "features": ["a"]}
        result = merge_settings(settings_path, new_settings)
        assert result == new_settings

    def test_merge_existing_file(self, tmp_path: Path) -> None:
        """Merge with existing settings performs deep merge."""
        settings_path = tmp_path / "settings.json"
        settings_path.write_text('{"model": "sonnet", "features": ["a"]}')

        new_settings = {"features": ["b"], "timeout": 30}
        result = merge_settings(settings_path, new_settings)

        assert result["model"] == "sonnet"  # Unchanged
        assert result["features"] == ["a", "b"]  # Extended
        assert result["timeout"] == 30  # Added


class TestInstallComponent:
    """Test install_component function."""

    @pytest.fixture
    def skill_component(self) -> ComponentMetadata:
        """Create a test skill component."""
        return ComponentMetadata(
            id="test/repo/skill/example-skill",
            name="Example Skill",
            component_type=ComponentType.SKILL,
            raw_content="# Example Skill\n\nDoes something useful.",
        )

    @pytest.fixture
    def command_component(self) -> ComponentMetadata:
        """Create a test command component."""
        return ComponentMetadata(
            id="test/repo/command/deploy",
            name="Deploy Command",
            component_type=ComponentType.COMMAND,
            raw_content="# Deploy\n\nDeploys the app.",
        )

    @pytest.fixture
    def setting_component(self) -> ComponentMetadata:
        """Create a test setting component."""
        return ComponentMetadata(
            id="test/repo/setting/config",
            name="Config",
            component_type=ComponentType.SETTING,
            raw_content='{"model": "opus", "timeout": 60}',
        )

    def test_install_skill(self, tmp_path: Path, skill_component: ComponentMetadata) -> None:
        """Skill installed to correct path."""
        dest, tokens = install_component(skill_component, tmp_path)

        assert dest == tmp_path / ".claude/skills/example-skill/SKILL.md"
        assert dest.exists()
        assert dest.read_text() == skill_component.raw_content
        assert tokens > 0

    def test_install_command(self, tmp_path: Path, command_component: ComponentMetadata) -> None:
        """Command installed to correct path."""
        dest, _tokens = install_component(command_component, tmp_path)

        assert dest == tmp_path / ".claude/commands/deploy-command.md"
        assert dest.exists()
        assert dest.read_text() == command_component.raw_content

    def test_install_setting_creates_json(
        self, tmp_path: Path, setting_component: ComponentMetadata
    ) -> None:
        """Setting installed as settings.json."""
        dest, _tokens = install_component(setting_component, tmp_path)

        assert dest == tmp_path / ".claude/settings.json"
        assert dest.exists()

        content = json.loads(dest.read_text())
        assert content["model"] == "opus"
        assert content["timeout"] == 60

    def test_install_setting_merges(
        self, tmp_path: Path, setting_component: ComponentMetadata
    ) -> None:
        """Installing setting merges with existing settings.json."""
        # Create existing settings
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir(parents=True)
        settings_path = settings_dir / "settings.json"
        settings_path.write_text('{"existing": true, "model": "sonnet"}')

        dest, _tokens = install_component(setting_component, tmp_path)

        content = json.loads(dest.read_text())
        assert content["existing"] is True  # Preserved
        assert content["model"] == "opus"  # Overwritten by new setting
        assert content["timeout"] == 60  # Added

    def test_creates_parent_directories(
        self, tmp_path: Path, skill_component: ComponentMetadata
    ) -> None:
        """Installation creates parent directories if needed."""
        dest, _ = install_component(skill_component, tmp_path)
        assert dest.parent.exists()

    def test_name_sanitization(self, tmp_path: Path) -> None:
        """Component name is sanitized for filesystem."""
        comp = ComponentMetadata(
            id="test/repo/skill/my-cool-skill",
            name="My Cool Skill",  # Spaces should become hyphens
            component_type=ComponentType.SKILL,
            raw_content="content",
        )
        dest, _ = install_component(comp, tmp_path)
        assert "my-cool-skill" in str(dest)


class TestComponentInstaller:
    """Test ComponentInstaller class."""

    @pytest.fixture
    def graph_store(self) -> NetworkXGraphStore:
        """Create graph store with test nodes and edges."""
        store = NetworkXGraphStore()

        # Add nodes
        store.add_node(
            GraphNode(
                id="test/repo/skill/auth",
                component_type=ComponentType.SKILL,
                label="Auth Skill",
            )
        )
        store.add_node(
            GraphNode(
                id="test/repo/skill/jwt",
                component_type=ComponentType.SKILL,
                label="JWT Skill",
            )
        )
        store.add_node(
            GraphNode(
                id="test/repo/skill/crypto",
                component_type=ComponentType.SKILL,
                label="Crypto Skill",
            )
        )

        # Add dependency: auth -> jwt -> crypto
        store.add_edge(
            GraphEdge(
                source_id="test/repo/skill/auth",
                target_id="test/repo/skill/jwt",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        store.add_edge(
            GraphEdge(
                source_id="test/repo/skill/jwt",
                target_id="test/repo/skill/crypto",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )

        return store

    @pytest.fixture
    def metadata_store(self, tmp_path: Path) -> MetadataStore:
        """Create metadata store with test components."""
        store = MetadataStore(tmp_path / "metadata.json")

        store.add(
            ComponentMetadata(
                id="test/repo/skill/auth",
                name="Auth Skill",
                component_type=ComponentType.SKILL,
                raw_content="# Auth\n\nAuthentication skill.",
            )
        )
        store.add(
            ComponentMetadata(
                id="test/repo/skill/jwt",
                name="JWT Skill",
                component_type=ComponentType.SKILL,
                raw_content="# JWT\n\nJWT handling.",
            )
        )
        store.add(
            ComponentMetadata(
                id="test/repo/skill/crypto",
                name="Crypto Skill",
                component_type=ComponentType.SKILL,
                raw_content="# Crypto\n\nCryptography utilities.",
            )
        )

        return store

    def test_install_single_component(
        self,
        tmp_path: Path,
        graph_store: NetworkXGraphStore,
        metadata_store: MetadataStore,
    ) -> None:
        """Install a single component without dependencies."""
        installer = ComponentInstaller(
            graph_store=graph_store,
            metadata_store=metadata_store,
            target_dir=tmp_path,
        )

        report = installer.install(["test/repo/skill/crypto"], auto_resolve_deps=False)

        assert len(report.installed) == 1
        assert report.installed[0].success
        assert report.installed[0].component_id == "test/repo/skill/crypto"
        assert (tmp_path / ".claude/skills/crypto-skill/SKILL.md").exists()

    def test_install_with_auto_deps(
        self,
        tmp_path: Path,
        graph_store: NetworkXGraphStore,
        metadata_store: MetadataStore,
    ) -> None:
        """Install component with automatic dependency resolution."""
        installer = ComponentInstaller(
            graph_store=graph_store,
            metadata_store=metadata_store,
            target_dir=tmp_path,
        )

        report = installer.install(["test/repo/skill/auth"], auto_resolve_deps=True)

        # Should install auth + jwt + crypto (transitive deps)
        installed_ids = {r.component_id for r in report.installed if r.success}
        assert "test/repo/skill/auth" in installed_ids
        assert "test/repo/skill/jwt" in installed_ids
        assert "test/repo/skill/crypto" in installed_ids

    def test_install_missing_component_error(
        self,
        tmp_path: Path,
        graph_store: NetworkXGraphStore,
        metadata_store: MetadataStore,
    ) -> None:
        """Error when component not found in metadata store."""
        installer = ComponentInstaller(
            graph_store=graph_store,
            metadata_store=metadata_store,
            target_dir=tmp_path,
        )

        report = installer.install(["test/repo/skill/nonexistent"])

        assert len(report.errors) > 0
        assert "not found" in report.errors[0].lower()

    def test_conflict_blocks_installation(
        self,
        tmp_path: Path,
        graph_store: NetworkXGraphStore,
        metadata_store: MetadataStore,
    ) -> None:
        """Conflicts should block installation."""
        # Add conflicting node and edge
        graph_store.add_node(
            GraphNode(
                id="test/repo/skill/alt-jwt",
                component_type=ComponentType.SKILL,
                label="Alt JWT",
            )
        )
        graph_store.add_edge(
            GraphEdge(
                source_id="test/repo/skill/jwt",
                target_id="test/repo/skill/alt-jwt",
                edge_type=EdgeType.CONFLICTS_WITH,
                metadata={"reason": "Incompatible JWT implementations"},
            )
        )
        metadata_store.add(
            ComponentMetadata(
                id="test/repo/skill/alt-jwt",
                name="Alt JWT",
                component_type=ComponentType.SKILL,
                raw_content="# Alt JWT",
            )
        )

        installer = ComponentInstaller(
            graph_store=graph_store,
            metadata_store=metadata_store,
            target_dir=tmp_path,
        )

        # Try to install both conflicting components
        report = installer.install(
            ["test/repo/skill/jwt", "test/repo/skill/alt-jwt"],
            auto_resolve_deps=False,
        )

        assert len(report.errors) > 0
        assert any("conflict" in e.lower() for e in report.errors)
        # Should not have installed anything
        assert len(report.installed) == 0

    def test_reports_token_cost(
        self,
        tmp_path: Path,
        graph_store: NetworkXGraphStore,
        metadata_store: MetadataStore,
    ) -> None:
        """Installation report includes token costs."""
        installer = ComponentInstaller(
            graph_store=graph_store,
            metadata_store=metadata_store,
            target_dir=tmp_path,
        )

        report = installer.install(["test/repo/skill/auth"], auto_resolve_deps=True)

        assert report.total_tokens > 0
        for item in report.installed:
            if item.success:
                assert item.token_cost > 0
