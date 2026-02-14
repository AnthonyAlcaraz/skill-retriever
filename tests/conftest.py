"""Shared test fixtures for skill-retriever."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def davila7_repo() -> Path:
    """Path to the davila7-style test fixture repository."""
    return FIXTURES_DIR / "davila7_sample"


@pytest.fixture
def flat_repo() -> Path:
    """Path to the flat .claude/ test fixture repository."""
    return FIXTURES_DIR / "flat_sample"


@pytest.fixture
def npm_repo(tmp_path: Path) -> Path:
    """Path to an npm MCP server test fixture."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "@anthropic/claude-code-mcp",
        "version": "1.0.0",
        "description": "MCP server for Claude Code",
        "keywords": ["mcp", "claude", "ai"],
        "dependencies": {
            "@modelcontextprotocol/sdk": "^1.0.0",
            "express": "^4.18.0",
        },
    }))
    (tmp_path / "README.md").write_text(
        "# claude-code-mcp\n\nAn MCP server for Claude Code integration.\n"
    )
    return tmp_path


@pytest.fixture
def npm_monorepo(tmp_path: Path) -> Path:
    """Path to an npm monorepo test fixture."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "mono",
        "private": True,
        "workspaces": ["packages/*"],
    }))
    for pkg_name in ("a", "b"):
        pkg_dir = tmp_path / "packages" / pkg_name
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "package.json").write_text(json.dumps({
            "name": f"@mono/{pkg_name}",
            "version": "1.0.0",
            "description": f"Package {pkg_name}",
            "keywords": ["skill"],
        }))
    return tmp_path


@pytest.fixture
def npm_cli_repo(tmp_path: Path) -> Path:
    """Path to an npm CLI tool test fixture."""
    (tmp_path / "package.json").write_text(json.dumps({
        "name": "sweetlink",
        "version": "2.0.0",
        "description": "A fast URL shortener CLI",
        "keywords": ["cli", "url", "shortener"],
        "bin": {"sweetlink": "./bin/sweetlink.js"},
    }))
    return tmp_path


@pytest.fixture
def readme_only_repo(tmp_path: Path) -> Path:
    """Path to a repo with only a README (no package.json, no markdown components)."""
    (tmp_path / "README.md").write_text(
        "# Oracle\n\n"
        "A CLI tool for querying local LLM models from the terminal.\n\n"
        "## Installation\n\n"
        "```bash\nbrew install oracle\n```\n"
    )
    return tmp_path


@pytest.fixture
def readme_mcp_repo(tmp_path: Path) -> Path:
    """Path to a repo whose README indicates it is an MCP server."""
    (tmp_path / "README.md").write_text(
        "# My MCP Server\n\n"
        "A Model Context Protocol server for file operations.\n\n"
        "## Features\n\n- Read files\n- Write files\n"
    )
    return tmp_path
