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


class PluginMarketplaceStrategy:
    """Strategy for plugin marketplace repos (e.g., zxkane/aws-skills, obra/superpowers-marketplace).

    Expects: ``plugins/{plugin-name}/skills/{skill-name}/SKILL.md``
    or: ``plugins/{plugin-name}/agents/{agent-name}/AGENT.md``
    """

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:
        plugins_dir = repo_root / "plugins"
        if not plugins_dir.is_dir():
            return False
        # Check if any plugin has skills/ or agents/ subdirectory
        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir():
                if (plugin_dir / "skills").is_dir() or (plugin_dir / "agents").is_dir():
                    return True
        return False

    def discover(self, repo_root: Path) -> list[Path]:
        plugins_dir = repo_root / "plugins"
        files: list[Path] = []
        for plugin_dir in plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue
            # Look for skills
            skills_dir = plugin_dir / "skills"
            if skills_dir.is_dir():
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir():
                        skill_md = skill_dir / "SKILL.md"
                        if skill_md.exists():
                            files.append(skill_md)
            # Look for agents
            agents_dir = plugin_dir / "agents"
            if agents_dir.is_dir():
                for agent_dir in agents_dir.iterdir():
                    if agent_dir.is_dir():
                        # Try AGENT.md first, then fallback to any .md
                        agent_md = agent_dir / "AGENT.md"
                        if agent_md.exists():
                            files.append(agent_md)
                        else:
                            # Look for any markdown file with name frontmatter
                            for md_file in agent_dir.glob("*.md"):
                                raw_meta, _ = parse_component_file(md_file)
                                if raw_meta.get("name"):
                                    files.append(md_file)
                                    break
        return sorted(files)

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        raw_meta, content = parse_component_file(file_path)
        meta = normalize_frontmatter(raw_meta)

        # Get name from frontmatter or directory name
        name = meta.get("name")
        if not name:
            name = file_path.parent.name  # Use directory name as fallback

        if not name:
            return None

        # Infer type from path structure
        parts_lower = [p.lower() for p in file_path.parts]
        if "agents" in parts_lower or file_path.name.upper() == "AGENT.MD":
            component_type = ComponentType.AGENT
        elif "skills" in parts_lower or file_path.name.upper() == "SKILL.MD":
            component_type = ComponentType.SKILL
        else:
            component_type = _infer_type_from_path(file_path)

        # Extract plugin name for category
        try:
            rel = file_path.relative_to(repo_root / "plugins")
            plugin_name = rel.parts[0] if rel.parts else ""
        except ValueError:
            plugin_name = ""

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
            category=plugin_name,
        )


class AwesomeListStrategy:
    """Strategy for curated awesome-list repos that link to external skills.

    Parses README.md to extract skill references and metadata.
    Extracts: name, description, URL, tags from markdown lists/tables.
    """

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:
        # Check if repo name suggests awesome list
        if "awesome" not in self.repo_name.lower():
            return False
        readme = repo_root / "README.md"
        if not readme.exists():
            return False
        # Check if README contains skill/component links
        try:
            content = readme.read_text(encoding="utf-8")
            # Look for patterns indicating a curated list
            return (
                "github.com" in content.lower()
                and ("skill" in content.lower() or "agent" in content.lower())
                and ("-" in content or "*" in content)  # List markers
            )
        except Exception:
            return False

    def discover(self, repo_root: Path) -> list[Path]:
        # For awesome lists, we return the README as the source file
        readme = repo_root / "README.md"
        return [readme] if readme.exists() else []

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        # This strategy returns multiple components from a single file
        # For now, just mark the README as processed and return None
        # The actual extraction happens in extract_all()
        return None

    def extract_all(self, repo_root: Path) -> list[ComponentMetadata]:
        """Extract all components from the awesome list README."""
        import re

        readme = repo_root / "README.md"
        if not readme.exists():
            return []

        try:
            content = readme.read_text(encoding="utf-8")
        except Exception:
            return []

        components: list[ComponentMetadata] = []

        # Pattern 1: Markdown links with descriptions
        # [name](url) - description
        # **[name](url)** - description
        link_pattern = re.compile(
            r'\*?\*?\[([^\]]+)\]\((https?://[^\)]+)\)\*?\*?\s*[-–—:]\s*(.+?)(?:\n|$)',
            re.IGNORECASE
        )

        for match in link_pattern.finditer(content):
            name = match.group(1).strip()
            url = match.group(2).strip()
            description = match.group(3).strip()

            # Skip non-skill links
            if not any(kw in url.lower() for kw in ["github.com", "skill", "agent", "claude"]):
                continue

            # Infer type from context
            context_start = max(0, match.start() - 200)
            context = content[context_start:match.start()].lower()
            if "agent" in context:
                component_type = ComponentType.AGENT
            elif "hook" in context:
                component_type = ComponentType.HOOK
            elif "mcp" in context:
                component_type = ComponentType.MCP
            else:
                component_type = ComponentType.SKILL

            # Extract tags from description
            tags = []
            tag_matches = re.findall(r'`([^`]+)`', description)
            tags.extend(tag_matches[:5])

            # Generate unique ID
            safe_name = re.sub(r'[^a-z0-9-]', '-', name.lower())[:50]
            component_id = ComponentMetadata.generate_id(
                self.repo_owner, self.repo_name, component_type, safe_name
            )

            components.append(ComponentMetadata(
                id=component_id,
                name=name,
                component_type=component_type,
                description=description[:500],
                tags=tags,
                tools=[],
                dependencies=[],
                version="",
                raw_content=f"[{name}]({url})\n{description}",
                source_repo=f"{self.repo_owner}/{self.repo_name}",
                source_path="README.md",
                category="curated",
                install_url=url if "github.com" in url else None,
            ))

        return components


class PackageJsonStrategy:
    """Strategy for npm/TypeScript repos with package.json.

    Extracts one component per package.json found. Handles monorepos
    by discovering packages/*/package.json and apps/*/package.json.
    Infers ComponentType from package.json keywords and dependencies.
    Uses README.md content as raw_content for richer embeddings.
    """

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:
        return (repo_root / "package.json").is_file()

    def discover(self, repo_root: Path) -> list[Path]:
        files: list[Path] = []
        root_pkg = repo_root / "package.json"
        if root_pkg.is_file():
            files.append(root_pkg)
        # Monorepo: packages/*/package.json
        for subdir_name in ("packages", "apps"):
            subdir = repo_root / subdir_name
            if subdir.is_dir():
                for child in subdir.iterdir():
                    if child.is_dir():
                        pkg = child / "package.json"
                        if pkg.is_file():
                            files.append(pkg)
        return sorted(files)

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        import json as _json

        try:
            data = _json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        name = data.get("name", "")
        if not name:
            name = file_path.parent.name
        if not name:
            return None

        # Clean scoped package names: @scope/name -> name
        if "/" in name:
            name = name.split("/")[-1]

        description = data.get("description", "")
        keywords = data.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]

        deps_dict = data.get("dependencies", {})
        dev_deps_dict = data.get("devDependencies", {})
        all_dep_names = sorted(set(list(deps_dict.keys()) + list(dev_deps_dict.keys())))

        component_type = self._infer_type(keywords, all_dep_names, name, description)
        tags = [k.lower() for k in keywords[:10]]

        # Read README for raw_content (richer embeddings)
        raw_content = ""
        pkg_dir = file_path.parent
        for readme_name in ("README.md", "readme.md", "README.MD", "Readme.md"):
            readme_path = pkg_dir / readme_name
            if readme_path.is_file():
                try:
                    raw_content = readme_path.read_text(encoding="utf-8")
                except Exception:
                    pass
                break

        # If no local README, try repo root (only for root package.json)
        if not raw_content and file_path.parent == repo_root:
            for readme_name in ("README.md", "readme.md"):
                readme_path = repo_root / readme_name
                if readme_path.is_file():
                    try:
                        raw_content = readme_path.read_text(encoding="utf-8")
                    except Exception:
                        pass
                    break

        if not raw_content:
            raw_content = description

        # Category: for monorepo sub-packages, use the parent dir path
        category = ""
        try:
            rel = file_path.relative_to(repo_root)
            parts = list(rel.parts[:-1])  # Remove "package.json"
            if parts:
                category = "/".join(parts)
        except ValueError:
            pass

        component_id = ComponentMetadata.generate_id(
            self.repo_owner, self.repo_name, component_type, name
        )

        return ComponentMetadata(
            id=component_id,
            name=name,
            component_type=component_type,
            description=description[:500],
            tags=tags,
            tools=[],
            dependencies=all_dep_names[:20],
            version=str(data.get("version", "")),
            raw_content=raw_content,
            source_repo=f"{self.repo_owner}/{self.repo_name}",
            source_path=str(file_path.relative_to(repo_root)),
            category=category,
        )

    @staticmethod
    def _infer_type(
        keywords: list[str],
        dep_names: list[str],
        name: str,
        description: str,
    ) -> ComponentType:
        """Infer ComponentType from package.json signals."""
        kw_lower = [k.lower() for k in keywords]
        name_lower = name.lower()
        desc_lower = description.lower()

        # MCP detection (highest priority)
        if "mcp" in kw_lower or "mcp-server" in kw_lower:
            return ComponentType.MCP
        if "mcp" in name_lower or "model-context-protocol" in desc_lower:
            return ComponentType.MCP
        if any("@modelcontextprotocol" in d for d in dep_names):
            return ComponentType.MCP

        # Agent detection
        if "agent" in kw_lower or "autonomous" in kw_lower:
            return ComponentType.AGENT
        if "agent" in name_lower:
            return ComponentType.AGENT

        # Hook detection
        if "hook" in kw_lower or "git-hook" in kw_lower:
            return ComponentType.HOOK

        # Command/CLI detection
        if "cli" in kw_lower or "command" in kw_lower:
            return ComponentType.COMMAND
        if name_lower.endswith("-cli"):
            return ComponentType.COMMAND

        return ComponentType.SKILL


class ReadmeFallbackStrategy:
    """Catch-all strategy for repos where no other strategy produced components.

    Parses README.md to extract a single component representing the repo.
    Uses the first H1 as name and first paragraph as description.
    MUST be the last strategy tried.
    """

    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def can_handle(self, repo_root: Path) -> bool:
        for name in ("README.md", "readme.md", "README.MD", "Readme.md"):
            if (repo_root / name).is_file():
                return True
        return False

    def discover(self, repo_root: Path) -> list[Path]:
        for name in ("README.md", "readme.md", "README.MD", "Readme.md"):
            readme = repo_root / name
            if readme.is_file():
                return [readme]
        return []

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata | None:
        import re as _re

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return None

        if not content.strip():
            return None

        # Extract name from first H1 heading, fallback to repo name
        name = self.repo_name
        h1_match = _re.search(r'^#\s+(.+?)$', content, _re.MULTILINE)
        if h1_match:
            raw_name = h1_match.group(1).strip()
            # Remove badges, images, links
            raw_name = _re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', raw_name)
            raw_name = _re.sub(r'!\[.*?\]\(.*?\)', '', raw_name)
            raw_name = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', raw_name)
            raw_name = raw_name.strip()
            if raw_name:
                name = raw_name

        # Extract description from first non-heading, non-empty paragraph
        description = ""
        paragraph_lines: list[str] = []
        in_frontmatter = False
        in_paragraph = False
        for line in content.split("\n"):
            stripped = line.strip()
            # Skip YAML frontmatter
            if stripped == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                continue
            # Skip headings, badges
            if stripped.startswith("#") or stripped.startswith("![") or stripped.startswith("[!["):
                if in_paragraph:
                    break
                continue
            if not stripped:
                if in_paragraph:
                    break
                continue
            in_paragraph = True
            paragraph_lines.append(stripped)

        if paragraph_lines:
            description = " ".join(paragraph_lines)

        if not description:
            description = f"Component from {self.repo_owner}/{self.repo_name}"

        component_type = self._infer_type(name, description, content)

        component_id = ComponentMetadata.generate_id(
            self.repo_owner, self.repo_name, component_type, name
        )

        return ComponentMetadata(
            id=component_id,
            name=name,
            component_type=component_type,
            description=description[:500],
            tags=self._extract_tags(content),
            tools=[],
            dependencies=[],
            version="",
            raw_content=content,
            source_repo=f"{self.repo_owner}/{self.repo_name}",
            source_path=str(file_path.relative_to(repo_root)),
            category="",
        )

    @staticmethod
    def _infer_type(name: str, description: str, content: str) -> ComponentType:
        """Infer ComponentType from README content signals."""
        all_text = f"{name} {description}".lower()
        content_sample = content[:2000].lower()

        if "mcp" in all_text or "model context protocol" in content_sample:
            return ComponentType.MCP
        if "agent" in all_text or "autonomous" in all_text:
            return ComponentType.AGENT
        if "hook" in all_text or "git hook" in content_sample:
            return ComponentType.HOOK
        if "cli" in all_text or "command line" in all_text or "command-line" in all_text:
            return ComponentType.COMMAND
        return ComponentType.SKILL

    @staticmethod
    def _extract_tags(content: str) -> list[str]:
        """Extract tags from README badges and H2 headings."""
        import re as _re

        tags: list[str] = []
        _generic_headings = {
            "installation", "usage", "license", "contributing",
            "getting started", "table of contents", "requirements",
            "setup", "development", "changelog", "credits", "authors",
        }

        # Extract from shields.io badge URLs
        for badge in _re.findall(r'img\.shields\.io/badge/([^-]+)', content)[:5]:
            tag = badge.lower().replace("%20", " ").strip()
            if tag and tag not in tags:
                tags.append(tag)

        # Extract from H2 headings as topic indicators
        for heading in _re.findall(r'^##\s+(.+?)$', content, _re.MULTILINE)[:5]:
            tag = heading.strip().lower()
            if tag not in _generic_headings and tag and tag not in tags:
                tags.append(tag)

        return tags[:10]
