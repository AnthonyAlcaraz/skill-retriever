"""Extraction strategies for discovering components in repository layouts."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import Protocol, runtime_checkable

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
            dependencies=meta.get("dependencies", []),
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
            dependencies=meta.get("dependencies", []),
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

    def can_handle(self, repo_root: Path) -> bool:
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
            dependencies=meta.get("dependencies", []),
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


class PythonModuleStrategy:
    """Strategy for extracting components from Python source files.

    Extracts modules, classes, and functions with docstrings as components.
    Maps to component types based on naming conventions:
    - *_agent.py, agents/ → AGENT
    - *_mcp.py, mcp/ → MCP
    - *_hook.py, hooks/ → HOOK
    - Otherwise → SKILL
    """

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:
        """Check if repo has Python source files."""
        # Look for src/ or any .py files
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            return any(src_dir.rglob("*.py"))
        return any(repo_root.glob("*.py"))

    def discover(self, repo_root: Path) -> list[Path]:
        """Find all Python files, excluding tests and common non-component files."""
        files: list[Path] = []
        exclude_patterns = {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "test",
            "tests",
            "conftest.py",
            "setup.py",
            "__init__.py",
        }

        for py_file in repo_root.rglob("*.py"):
            # Skip excluded directories and files
            if any(part in exclude_patterns for part in py_file.parts):
                continue
            if py_file.name in exclude_patterns:
                continue
            # Only include files with docstrings (meaningful modules)
            try:
                content = py_file.read_text(encoding="utf-8")
                if '"""' in content or "'''" in content:
                    files.append(py_file)
            except Exception:
                continue

        return sorted(files)

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        """Extract component metadata from a Python file."""
        import ast

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except Exception:
            return None

        # Get module docstring
        module_doc = ast.get_docstring(tree) or ""

        # Extract name from filename
        name = file_path.stem

        # Infer component type from path/filename
        component_type = self._infer_python_type(file_path)

        # Build description from module docstring + class/function signatures
        description_parts = [module_doc] if module_doc else []

        # Extract class and function names for richer description
        classes: list[str] = []
        functions: list[str] = []
        dependencies: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("_"):
                        dependencies.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith("_"):
                    dependencies.append(node.module.split(".")[0])

        if classes:
            description_parts.append(f"Classes: {', '.join(classes[:5])}")
        if functions:
            # Filter out private functions
            public_funcs = [f for f in functions if not f.startswith("_")][:5]
            if public_funcs:
                description_parts.append(f"Functions: {', '.join(public_funcs)}")

        description = ". ".join(description_parts) if description_parts else name

        # Deduplicate dependencies
        dependencies = sorted(set(dependencies))

        component_id = ComponentMetadata.generate_id(
            self.repo_owner, self.repo_name, component_type, name
        )

        return ComponentMetadata(
            id=component_id,
            name=name,
            component_type=component_type,
            description=description[:500],  # Truncate long descriptions
            tags=self._extract_tags(file_path, classes),
            tools=[],
            dependencies=dependencies,
            version="",
            raw_content=content,
            source_repo=f"{self.repo_owner}/{self.repo_name}",
            source_path=str(file_path.relative_to(repo_root)),
            category=self._extract_category(file_path, repo_root),
        )

    def _infer_python_type(self, file_path: Path) -> ComponentType:
        """Infer component type from Python file path/name."""
        path_str = str(file_path).lower()
        name = file_path.stem.lower()

        # Check filename patterns
        if name.endswith("_agent") or "agent" in name:
            return ComponentType.AGENT
        if name.endswith("_mcp") or "mcp" in path_str:
            return ComponentType.MCP
        if name.endswith("_hook") or "hook" in path_str:
            return ComponentType.HOOK
        if "command" in name or "cli" in name:
            return ComponentType.COMMAND
        if "config" in name or "setting" in name:
            return ComponentType.SETTING

        # Check directory patterns
        parts = [p.lower() for p in file_path.parts]
        if "agents" in parts:
            return ComponentType.AGENT
        if "mcp" in parts:
            return ComponentType.MCP
        if "hooks" in parts:
            return ComponentType.HOOK
        if "commands" in parts:
            return ComponentType.COMMAND

        return ComponentType.SKILL

    def _extract_tags(self, file_path: Path, classes: list[str]) -> list[str]:
        """Extract tags from file path and class names."""
        tags: list[str] = []

        # Add directory-based tags
        for part in file_path.parts:
            if part not in {"src", "lib", "app", ".py"} and not part.startswith("_"):
                tags.append(part.lower())

        # Add class-based tags (simplified)
        for cls in classes[:3]:
            # Convert CamelCase to tag
            tag = "".join(
                f"-{c.lower()}" if c.isupper() else c for c in cls
            ).lstrip("-")
            if tag and tag not in tags:
                tags.append(tag)

        return tags[:10]  # Limit tags

    def _extract_category(self, file_path: Path, repo_root: Path) -> str:
        """Extract category from relative path."""
        try:
            rel = file_path.relative_to(repo_root)
            # Get intermediate directories (excluding src/ and filename)
            parts = list(rel.parts[:-1])
            if parts and parts[0] == "src":
                parts = parts[1:]
            if parts and parts[0].replace("_", "-") == self.repo_name.replace("_", "-"):
                parts = parts[1:]
            return "/".join(parts) if parts else ""
        except ValueError:
            return ""
