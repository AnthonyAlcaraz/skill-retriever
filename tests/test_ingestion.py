"""Tests for the ingestion pipeline: frontmatter, git_signals, extractors, crawler."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from skill_retriever.entities import ComponentType
from skill_retriever.nodes.ingestion.crawler import RepositoryCrawler
from skill_retriever.nodes.ingestion.extractors import (
    Davila7Strategy,
    FlatDirectoryStrategy,
)
from skill_retriever.nodes.ingestion.frontmatter import (
    normalize_frontmatter,
    parse_component_file,
)
from skill_retriever.nodes.ingestion.git_signals import extract_git_signals

# ---------------------------------------------------------------------------
# Frontmatter tests
# ---------------------------------------------------------------------------


def test_parse_component_file_with_frontmatter(davila7_repo: Path) -> None:
    """Parsing a file with YAML frontmatter returns metadata and body."""
    agent_file = (
        davila7_repo
        / "cli-tool"
        / "components"
        / "agents"
        / "ai-specialists"
        / "prompt-engineer.md"
    )
    meta, body = parse_component_file(agent_file)
    assert meta["name"] == "prompt-engineer"
    assert meta["description"] == "Expert at crafting effective prompts for AI systems"
    assert meta["tools"] == ["Read", "Write", "Edit"]
    assert "## Expertise Areas" in body


def test_parse_component_file_no_frontmatter(tmp_path: Path) -> None:
    """A markdown file without frontmatter returns empty metadata."""
    plain = tmp_path / "plain.md"
    plain.write_text("# Just a heading\nNo frontmatter here.\n")
    meta, body = parse_component_file(plain)
    assert meta == {}
    assert "Just a heading" in body


def test_normalize_frontmatter_allowed_tools() -> None:
    """The 'allowed-tools' key is mapped to 'tools'."""
    raw = {"name": "test", "allowed-tools": ["Bash", "Read"]}
    result = normalize_frontmatter(raw)
    assert result["tools"] == ["Bash", "Read"]
    assert "allowed-tools" not in result


def test_normalize_frontmatter_string_tags() -> None:
    """A comma-separated tags string is split into a list."""
    raw = {"name": "test", "tags": "python, ai, tools"}
    result = normalize_frontmatter(raw)
    assert result["tags"] == ["python", "ai", "tools"]


# ---------------------------------------------------------------------------
# Git signals tests
# ---------------------------------------------------------------------------


def test_extract_git_signals_no_git(tmp_path: Path) -> None:
    """Non-git directory returns default signals."""
    signals = extract_git_signals(tmp_path, "some/file.md")
    assert signals["last_updated"] is None
    assert signals["commit_count"] == 0
    assert signals["commit_frequency_30d"] == 0.0


# ---------------------------------------------------------------------------
# Davila7 strategy tests
# ---------------------------------------------------------------------------


def test_davila7_strategy_can_handle(davila7_repo: Path) -> None:
    """Davila7Strategy recognizes cli-tool/components/ layout."""
    strategy = Davila7Strategy("davila7", "claude-code-cli")
    assert strategy.can_handle(davila7_repo) is True


def test_davila7_strategy_cannot_handle_flat(flat_repo: Path) -> None:
    """Davila7Strategy rejects repos without cli-tool/components/."""
    strategy = Davila7Strategy("owner", "repo")
    assert strategy.can_handle(flat_repo) is False


def test_davila7_strategy_discover(davila7_repo: Path) -> None:
    """Davila7Strategy discovers all markdown files under component type dirs."""
    strategy = Davila7Strategy("davila7", "claude-code-cli")
    files = strategy.discover(davila7_repo)
    assert len(files) == 3
    names = {f.name for f in files}
    assert "prompt-engineer.md" in names
    assert "SKILL.md" in names
    assert "pre-commit.md" in names


def test_davila7_strategy_extract_agent(davila7_repo: Path) -> None:
    """Davila7Strategy correctly extracts an agent component."""
    strategy = Davila7Strategy("davila7", "claude-code-cli")
    agent_file = (
        davila7_repo
        / "cli-tool"
        / "components"
        / "agents"
        / "ai-specialists"
        / "prompt-engineer.md"
    )
    component = strategy.extract(agent_file, davila7_repo)
    assert component is not None
    assert component.name == "prompt-engineer"
    assert component.component_type == ComponentType.AGENT
    assert component.category == "ai-specialists"
    assert component.tools == ["Read", "Write", "Edit"]
    assert component.id == "davila7/claude-code-cli/agent/prompt-engineer"


# ---------------------------------------------------------------------------
# Flat directory strategy tests
# ---------------------------------------------------------------------------


def test_flat_strategy_can_handle(flat_repo: Path) -> None:
    """FlatDirectoryStrategy recognizes .claude/ with recognized subdirs."""
    strategy = FlatDirectoryStrategy("owner", "repo")
    assert strategy.can_handle(flat_repo) is True


def test_flat_strategy_discover(flat_repo: Path) -> None:
    """FlatDirectoryStrategy discovers markdown files in .claude/{type}/."""
    strategy = FlatDirectoryStrategy("owner", "repo")
    files = strategy.discover(flat_repo)
    assert len(files) == 2
    names = {f.name for f in files}
    assert "code-reviewer.md" in names
    assert "deploy.md" in names


def test_flat_strategy_extract(flat_repo: Path) -> None:
    """FlatDirectoryStrategy correctly extracts a component."""
    strategy = FlatDirectoryStrategy("owner", "repo")
    agent_file = flat_repo / ".claude" / "agents" / "code-reviewer.md"
    component = strategy.extract(agent_file, flat_repo)
    assert component is not None
    assert component.name == "code-reviewer"
    assert component.component_type == ComponentType.AGENT
    assert component.tools == ["Read", "Grep"]
    assert component.id == "owner/repo/agent/code-reviewer"


# ---------------------------------------------------------------------------
# Crawler integration tests
# ---------------------------------------------------------------------------


def test_crawler_davila7(davila7_repo: Path) -> None:
    """RepositoryCrawler extracts all components from a davila7-style repo."""
    crawler = RepositoryCrawler("davila7", "claude-code-cli", davila7_repo)
    components = crawler.crawl()
    assert len(components) == 3
    types = {c.component_type for c in components}
    assert ComponentType.AGENT in types
    assert ComponentType.SKILL in types
    assert ComponentType.HOOK in types


def test_crawler_flat(flat_repo: Path) -> None:
    """RepositoryCrawler extracts all components from a flat .claude/ repo."""
    crawler = RepositoryCrawler("owner", "repo", flat_repo)
    components = crawler.crawl()
    assert len(components) == 2
    types = {c.component_type for c in components}
    assert ComponentType.AGENT in types
    assert ComponentType.COMMAND in types


def test_crawler_component_ids_are_deterministic(davila7_repo: Path) -> None:
    """Running the crawler twice produces identical component IDs."""
    crawler = RepositoryCrawler("davila7", "claude-code-cli", davila7_repo)
    first_run = crawler.crawl()
    second_run = crawler.crawl()
    first_ids = sorted(c.id for c in first_run)
    second_ids = sorted(c.id for c in second_run)
    assert first_ids == second_ids
    # Verify expected IDs
    assert "davila7/claude-code-cli/agent/prompt-engineer" in first_ids
    assert "davila7/claude-code-cli/skill/clean-code" in first_ids
    assert "davila7/claude-code-cli/hook/pre-commit" in first_ids
