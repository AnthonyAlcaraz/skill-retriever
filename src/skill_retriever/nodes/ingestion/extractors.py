"""Extraction strategies for discovering components in repository layouts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from skill_retriever.entities import ComponentMetadata, ComponentType
from skill_retriever.nodes.ingestion.frontmatter import (
    normalize_frontmatter,
    parse_component_file,
)

logger = logging.getLogger(__name__)

COMPONENT_TYPE_DIRS: dict[str, ComponentType] = {
    "agents": ComponentType.AGENT,
    "skills": ComponentType.SKILL,
    "commands": ComponentType.COMMAND,
    "hooks": ComponentType.HOOK,
    "settings": ComponentType.SETTING,
    "mcps": ComponentType.MCP,
    "sandbox": ComponentType.SANDBOX,
}

_EXCLUDED_DIRS = {".git", ".github", "node_modules", "__pycache__"}


@runtime_checkable
class ExtractionStrategy(Protocol):
    """Protocol for repository extraction strategies."""

    def can_handle(self, repo_root: Path) -> bool:
        """Return True if this strategy can handle the given repo layout."""
        ...

    def discover(self, repo_root: Path) -> list[Path]:
        """Return all component file paths found in the repo."""
        ...

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        """Extract component metadata from a single file."""
        ...


class Davila7Strategy:
    """Strategy for davila7/claude-code-cli style repos.

    Expects: ``cli-tool/components/{type_dir}/.../*.md``
    """

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:
        return (repo_root / "cli-tool" / "components").is_dir()

    def discover(self, repo_root: Path) -> list[Path]:
        components_dir = repo_root / "cli-tool" / "components"
        files: list[Path] = []
        for type_dir_name in COMPONENT_TYPE_DIRS:
            type_dir = components_dir / type_dir_name
            if type_dir.is_dir():
                files.extend(type_dir.rglob("*.md"))
        return sorted(files)

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        components_dir = repo_root / "cli-tool" / "components"
        try:
            rel = file_path.relative_to(components_dir)
        except ValueError:
            return None

        # First part of relative path is the type directory
        parts = rel.parts
        if not parts:
            return None

        type_dir_name = parts[0]
        component_type = COMPONENT_TYPE_DIRS.get(type_dir_name)
        if component_type is None:
            return None

        raw_meta, content = parse_component_file(file_path)
        meta = normalize_frontmatter(raw_meta)

        name = meta.get("name", file_path.stem)
        if not name:
            return None

        # Category from intermediate directories (between type dir and filename)
        category_parts = parts[1:-1]
        category = "/".join(category_parts) if category_parts else ""

        component_id = ComponentMetadata.generate_id(
            self.repo_owner, self.repo_name, component_type, name
        )

        return ComponentMetadata(
            id=component_id,
            name=name,
            component_type=component_type,
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            tools=meta.get("tools", []),
            version=meta.get("version", ""),
            raw_content=content,
            source_repo=f"{self.repo_owner}/{self.repo_name}",
            source_path=str(file_path.relative_to(repo_root)),
            category=category,
        )


class FlatDirectoryStrategy:
    """Strategy for repos with ``.claude/{type_dir}/`` layout."""

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:
        claude_dir = repo_root / ".claude"
        if not claude_dir.is_dir():
            return False
        # Check if any recognized subdirectory exists
        return any(
            (claude_dir / d).is_dir() for d in COMPONENT_TYPE_DIRS
        )

    def discover(self, repo_root: Path) -> list[Path]:
        claude_dir = repo_root / ".claude"
        files: list[Path] = []
        for type_dir_name in COMPONENT_TYPE_DIRS:
            type_dir = claude_dir / type_dir_name
            if type_dir.is_dir():
                files.extend(type_dir.glob("*.md"))
        return sorted(files)

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        claude_dir = repo_root / ".claude"
        try:
            rel = file_path.relative_to(claude_dir)
        except ValueError:
            return None

        parts = rel.parts
        if not parts:
            return None

        type_dir_name = parts[0]
        component_type = COMPONENT_TYPE_DIRS.get(type_dir_name)
        if component_type is None:
            return None

        raw_meta, content = parse_component_file(file_path)
        meta = normalize_frontmatter(raw_meta)

        name = meta.get("name", file_path.stem)
        if not name:
            return None

        component_id = ComponentMetadata.generate_id(
            self.repo_owner, self.repo_name, component_type, name
        )

        return ComponentMetadata(
            id=component_id,
            name=name,
            component_type=component_type,
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            tools=meta.get("tools", []),
            version=meta.get("version", ""),
            raw_content=content,
            source_repo=f"{self.repo_owner}/{self.repo_name}",
            source_path=str(file_path.relative_to(repo_root)),
            category="",
        )


class GenericMarkdownStrategy:
    """Fallback strategy: scans all markdown files for those with name in frontmatter."""

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:  # noqa: ARG002
        return True

    def discover(self, repo_root: Path) -> list[Path]:
        files: list[Path] = []
        for md_file in repo_root.rglob("*.md"):
            # Skip excluded directories
            if any(part in _EXCLUDED_DIRS for part in md_file.parts):
                continue
            # Only include files that have a name field in frontmatter
            raw_meta, _ = parse_component_file(md_file)
            if raw_meta.get("name"):
                files.append(md_file)
        return sorted(files)

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        raw_meta, content = parse_component_file(file_path)
        meta = normalize_frontmatter(raw_meta)

        name = meta.get("name")
        if not name:
            return None

        # Try to infer component type from path or default to skill
        component_type = _infer_type_from_path(file_path)

        component_id = ComponentMetadata.generate_id(
            self.repo_owner, self.repo_name, component_type, name
        )

        return ComponentMetadata(
            id=component_id,
            name=name,
            component_type=component_type,
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            tools=meta.get("tools", []),
            version=meta.get("version", ""),
            raw_content=content,
            source_repo=f"{self.repo_owner}/{self.repo_name}",
            source_path=str(file_path.relative_to(repo_root)),
            category="",
        )


def _infer_type_from_path(file_path: Path) -> ComponentType:
    """Infer component type from directory names in the file path."""
    parts_lower = [p.lower() for p in file_path.parts]
    for dir_name, comp_type in COMPONENT_TYPE_DIRS.items():
        if dir_name in parts_lower:
            return comp_type
    return ComponentType.SKILL
