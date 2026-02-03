"""Tests for MCP server tool registration and functionality."""

from __future__ import annotations

import pytest

from skill_retriever.entities.components import ComponentType
from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
from skill_retriever.mcp.rationale import EDGE_DESCRIPTIONS, generate_rationale, path_to_explanation
from skill_retriever.mcp.schemas import SearchInput, SearchResult
from skill_retriever.memory.graph_store import NetworkXGraphStore
from skill_retriever.nodes.retrieval.context_assembler import RetrievalContext
from skill_retriever.nodes.retrieval.models import RankedComponent


class TestToolRegistration:
    """Test that all 5 tools are registered."""

    def test_five_tools_registered(self) -> None:
        """MCP server should have exactly 5 tools registered."""
        from skill_retriever.mcp.server import mcp

        # Access internal tool manager to get tool count
        # pyright: ignore[reportPrivateUsage]
        tools = mcp._tool_manager._tools  # pyright: ignore[reportPrivateUsage]
        assert len(tools) == 5, f"Expected 5 tools, got {len(tools)}"

    def test_tool_names(self) -> None:
        """Verify expected tool names are registered."""
        from skill_retriever.mcp.server import mcp

        # pyright: ignore[reportPrivateUsage]
        tools = mcp._tool_manager._tools  # pyright: ignore[reportPrivateUsage]
        expected_names = {
            "search_components",
            "get_component_detail",
            "install_components",
            "check_dependencies",
            "ingest_repo",
        }
        actual_names = set(tools.keys())
        assert expected_names == actual_names, f"Expected {expected_names}, got {actual_names}"


class TestSchemasSerialization:
    """Test Pydantic model serialization."""

    def test_search_input_defaults(self) -> None:
        """SearchInput should have sensible defaults."""
        input_model = SearchInput(query="test query")
        assert input_model.query == "test query"
        assert input_model.top_k == 5
        assert input_model.component_type is None

    def test_search_input_with_type(self) -> None:
        """SearchInput should accept component_type."""
        input_model = SearchInput(
            query="authentication", top_k=10, component_type="skill"
        )
        assert input_model.component_type == "skill"
        assert input_model.top_k == 10

    def test_search_result_serialization(self) -> None:
        """SearchResult should serialize to dict."""
        from skill_retriever.mcp.schemas import ComponentRecommendation

        result = SearchResult(
            components=[
                ComponentRecommendation(
                    id="test/repo/skill/auth",
                    name="Auth Skill",
                    type="skill",
                    score=0.95,
                    rationale="Semantic match",
                    token_cost=150,
                )
            ],
            total_tokens=150,
            conflicts=[],
        )
        data = result.model_dump()
        assert len(data["components"]) == 1
        assert data["total_tokens"] == 150
        assert data["components"][0]["score"] == 0.95


class TestRationaleGeneration:
    """Test rationale generator with mock graph."""

    @pytest.fixture
    def graph_store(self) -> NetworkXGraphStore:
        """Create a graph store with test nodes and edges."""
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
                label="JWT Handler",
            )
        )
        store.add_node(
            GraphNode(
                id="test/repo/command/login",
                component_type=ComponentType.COMMAND,
                label="Login Command",
            )
        )

        # Add edges
        store.add_edge(
            GraphEdge(
                source_id="test/repo/command/login",
                target_id="test/repo/skill/auth",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
        store.add_edge(
            GraphEdge(
                source_id="test/repo/skill/auth",
                target_id="test/repo/skill/jwt",
                edge_type=EdgeType.ENHANCES,
            )
        )

        return store

    def test_edge_descriptions_complete(self) -> None:
        """All EdgeType values should have descriptions."""
        for edge_type in EdgeType:
            assert edge_type in EDGE_DESCRIPTIONS

    def test_path_to_explanation_direct_match(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Single-node path returns 'Direct match'."""
        result = path_to_explanation(["test/repo/skill/auth"], graph_store)
        assert result == "Direct match"

    def test_path_to_explanation_two_nodes(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Two-node path generates readable explanation."""
        result = path_to_explanation(
            ["test/repo/skill/auth", "test/repo/skill/jwt"], graph_store
        )
        assert "Auth Skill" in result
        assert "JWT Handler" in result
        # Should find the "enhances" edge
        assert "enhances" in result

    def test_generate_rationale_vector_source(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Vector-sourced components get 'Semantic match' rationale."""
        comp = RankedComponent(
            component_id="test/repo/skill/auth",
            score=0.9,
            rank=1,
            source="vector",
        )
        context = RetrievalContext(
            components=[comp], total_tokens=100, truncated=False, excluded_count=0
        )
        rationale = generate_rationale(comp, context, graph_store)
        assert rationale == "Semantic match to query"

    def test_generate_rationale_dependency_source(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Dependency-sourced components explain the parent."""
        auth_comp = RankedComponent(
            component_id="test/repo/skill/auth",
            score=0.1,
            rank=2,
            source="dependency",
        )
        login_comp = RankedComponent(
            component_id="test/repo/command/login",
            score=0.9,
            rank=1,
            source="vector",
        )
        context = RetrievalContext(
            components=[login_comp, auth_comp],
            total_tokens=200,
            truncated=False,
            excluded_count=0,
        )
        rationale = generate_rationale(auth_comp, context, graph_store)
        assert "dependency" in rationale.lower()

    def test_generate_rationale_graph_source(
        self, graph_store: NetworkXGraphStore
    ) -> None:
        """Graph-sourced components get relationship-based rationale."""
        comp = RankedComponent(
            component_id="test/repo/skill/auth",
            score=0.85,
            rank=1,
            source="graph",
        )
        context = RetrievalContext(
            components=[comp], total_tokens=100, truncated=False, excluded_count=0
        )
        rationale = generate_rationale(comp, context, graph_store)
        # Should mention the component or its relationships
        assert "Auth Skill" in rationale or "Graph" in rationale


class TestGitHubUrlParsing:
    """Test GitHub URL parsing in ingest_repo."""

    def test_https_url(self) -> None:
        """Parse standard HTTPS URL."""
        from skill_retriever.mcp.server import (
            _parse_github_url,  # pyright: ignore[reportPrivateUsage]
        )

        owner, name = _parse_github_url("https://github.com/owner/repo")
        assert owner == "owner"
        assert name == "repo"

    def test_https_url_with_git_suffix(self) -> None:
        """Parse HTTPS URL with .git suffix."""
        from skill_retriever.mcp.server import (
            _parse_github_url,  # pyright: ignore[reportPrivateUsage]
        )

        owner, name = _parse_github_url("https://github.com/owner/repo.git")
        assert owner == "owner"
        assert name == "repo"

    def test_ssh_url(self) -> None:
        """Parse SSH URL format."""
        from skill_retriever.mcp.server import (
            _parse_github_url,  # pyright: ignore[reportPrivateUsage]
        )

        owner, name = _parse_github_url("git@github.com:owner/repo.git")
        assert owner == "owner"
        assert name == "repo"

    def test_short_format(self) -> None:
        """Parse owner/repo shorthand."""
        from skill_retriever.mcp.server import (
            _parse_github_url,  # pyright: ignore[reportPrivateUsage]
        )

        owner, name = _parse_github_url("owner/repo")
        assert owner == "owner"
        assert name == "repo"

    def test_invalid_url_raises(self) -> None:
        """Invalid URL raises ValueError."""
        from skill_retriever.mcp.server import (
            _parse_github_url,  # pyright: ignore[reportPrivateUsage]
        )

        with pytest.raises(ValueError, match="Could not parse"):
            _parse_github_url("not-a-valid-url")


class TestInstallComponentsTool:
    """Test install_components tool handler."""

    def test_install_returns_result_model(self) -> None:
        """install_components returns InstallResult model."""
        from skill_retriever.mcp.schemas import InstallResult

        # Verify model structure
        result = InstallResult(
            installed=["test/repo/skill/auth"],
            skipped=[],
            errors=[],
        )
        assert result.installed == ["test/repo/skill/auth"]

    def test_install_input_defaults(self) -> None:
        """InstallInput has sensible defaults."""
        from skill_retriever.mcp.schemas import InstallInput

        input_model = InstallInput(component_ids=["test/repo/skill/auth"])
        assert input_model.target_dir == "."
        assert input_model.component_ids == ["test/repo/skill/auth"]
