"""Component installation engine for placing components in .claude/ directories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from skill_retriever.entities.components import ComponentType
from skill_retriever.nodes.retrieval.context_assembler import estimate_tokens

if TYPE_CHECKING:
    from pathlib import Path

    from skill_retriever.entities.components import ComponentMetadata
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.memory.metadata_store import MetadataStore

# Mapping from ComponentType to installation path template
# {name} is replaced with sanitized component name (lowercase, hyphens)
INSTALL_PATHS: dict[ComponentType, str] = {
    ComponentType.SKILL: ".claude/skills/{name}/SKILL.md",
    ComponentType.COMMAND: ".claude/commands/{name}.md",
    ComponentType.AGENT: ".claude/agents/{name}.md",
    ComponentType.SETTING: ".claude/settings.json",
    ComponentType.HOOK: ".claude/hooks/{name}/hook.md",
    ComponentType.MCP: ".claude/mcp-servers/{name}/config.json",
    ComponentType.SANDBOX: ".claude/sandbox/{name}/sandbox.md",
}


def _sanitize_name(name: str) -> str:
    """Sanitize component name for filesystem use.

    Args:
        name: The component name to sanitize.

    Returns:
        Lowercase name with spaces replaced by hyphens.
    """
    return re.sub(r"\s+", "-", name.strip()).lower()


def deep_merge(
    base: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge overlay dict into base dict.

    For dict values: recurse
    For list values: extend and dedupe
    For other values: overlay wins

    Args:
        base: The base dictionary to merge into.
        overlay: The overlay dictionary to merge from.

    Returns:
        New merged dictionary (does not mutate inputs).
    """
    result: dict[str, Any] = dict(base)

    for key, overlay_value in overlay.items():
        if key not in result:
            # Key only in overlay
            result[key] = overlay_value
        elif isinstance(result[key], dict) and isinstance(overlay_value, dict):
            # Both are dicts: recurse (narrow via isinstance)
            result[key] = deep_merge(
                result[key],  # pyright: ignore[reportArgumentType]
                overlay_value,  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]
            )
        elif isinstance(result[key], list) and isinstance(overlay_value, list):
            # Both are lists: extend and dedupe while preserving order
            combined: list[Any] = list(result[key])
            item: Any
            for item in overlay_value:  # pyright: ignore[reportUnknownVariableType]
                if item not in combined:
                    combined.append(item)
            result[key] = combined
        else:
            # Overlay wins for other types or mismatched types
            result[key] = overlay_value

    return result


def merge_settings(
    existing_path: Path, new_settings: dict[str, Any]
) -> dict[str, Any]:
    """Merge new settings into existing settings.json.

    Args:
        existing_path: Path to existing settings.json (may not exist).
        new_settings: New settings to merge in.

    Returns:
        Merged settings dictionary.
    """
    from pathlib import Path

    existing: dict[str, Any] = {}
    if Path(existing_path).exists():
        existing = json.loads(Path(existing_path).read_text(encoding="utf-8"))

    return deep_merge(existing, new_settings)


def install_component(
    component: ComponentMetadata,
    target_dir: Path,
) -> tuple[Path, int]:
    """Install a single component to the target directory.

    Args:
        component: The ComponentMetadata to install.
        target_dir: The base directory to install to (usually project root).

    Returns:
        Tuple of (destination_path, token_cost).
    """
    from pathlib import Path

    # Get path template for this component type
    path_template = INSTALL_PATHS.get(component.component_type)
    if path_template is None:
        msg = f"Unknown component type: {component.component_type}"
        raise ValueError(msg)

    # Build destination path
    sanitized_name = _sanitize_name(component.name)

    if component.component_type == ComponentType.SETTING:
        # Settings don't use name in path
        dest_path = Path(target_dir) / path_template
    else:
        dest_path = Path(target_dir) / path_template.format(name=sanitized_name)

    # Create parent directories
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate token cost
    token_cost = estimate_tokens(component.raw_content)

    # Handle settings specially (merge instead of overwrite)
    if component.component_type == ComponentType.SETTING:
        try:
            new_settings: dict[str, Any] = json.loads(component.raw_content)
        except json.JSONDecodeError:
            new_settings = {}
        merged = merge_settings(dest_path, new_settings)
        dest_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    else:
        # Write raw content for all other types
        dest_path.write_text(component.raw_content, encoding="utf-8")

    return dest_path, token_cost


@dataclass
class InstallationReport:
    """Report for a single component installation."""

    component_id: str
    destination: str
    token_cost: int
    success: bool
    error: str = ""


@dataclass
class InstallReport:
    """Full installation report for all components."""

    installed: list[InstallationReport] = field(
        default_factory=lambda: []  # pyright: ignore[reportUnknownLambdaType]
    )
    skipped: list[str] = field(
        default_factory=lambda: []  # pyright: ignore[reportUnknownLambdaType]
    )
    errors: list[str] = field(
        default_factory=lambda: []  # pyright: ignore[reportUnknownLambdaType]
    )
    total_tokens: int = 0


class ComponentInstaller:
    """Component installation engine with dependency resolution.

    Uses MetadataStore for component lookup and GraphStore for
    dependency and conflict resolution.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        metadata_store: MetadataStore,
        target_dir: Path,
    ) -> None:
        """Initialize the installer.

        Args:
            graph_store: Graph store for dependency/conflict resolution.
            metadata_store: Metadata store for component lookup.
            target_dir: Base directory to install components to.
        """
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        self.target_dir = target_dir

    def install(
        self,
        component_ids: list[str],
        auto_resolve_deps: bool = True,
    ) -> InstallReport:
        """Install components with optional dependency resolution.

        Args:
            component_ids: List of component IDs to install.
            auto_resolve_deps: If True, automatically resolve and install dependencies.

        Returns:
            InstallReport with results of the installation.
        """
        from skill_retriever.workflows.dependency_resolver import (
            detect_conflicts,
            resolve_transitive_dependencies,
        )

        report = InstallReport()

        # Validate all requested components exist in metadata store
        missing = [cid for cid in component_ids if cid not in self.metadata_store]
        if missing:
            for cid in missing:
                report.errors.append(f"Component not found: {cid}")
            return report

        # Resolve dependencies if requested
        if auto_resolve_deps:
            all_ids, deps_added = resolve_transitive_dependencies(
                component_ids, self.graph_store
            )
            # Add dependency IDs to install list
            install_ids = list(all_ids)

            # Check if dependencies exist in metadata store
            missing_deps = [
                cid for cid in deps_added if cid not in self.metadata_store
            ]
            if missing_deps:
                for cid in missing_deps:
                    report.errors.append(f"Dependency not found: {cid}")
                # Skip missing deps but continue with available ones
                install_ids = [cid for cid in install_ids if cid not in missing_deps]
                report.skipped.extend(missing_deps)
        else:
            install_ids = component_ids

        # Detect conflicts before installation
        conflicts = detect_conflicts(set(install_ids), self.graph_store)
        if conflicts:
            for conflict in conflicts:
                report.errors.append(
                    f"Conflict: {conflict.component_a} conflicts with "
                    f"{conflict.component_b}: {conflict.reason}"
                )
            # Block installation if conflicts exist
            return report

        # Install each component
        for component_id in install_ids:
            metadata = self.metadata_store.get(component_id)
            if metadata is None:
                report.skipped.append(component_id)
                continue

            try:
                dest_path, token_cost = install_component(metadata, self.target_dir)
                report.installed.append(
                    InstallationReport(
                        component_id=component_id,
                        destination=str(dest_path),
                        token_cost=token_cost,
                        success=True,
                    )
                )
                report.total_tokens += token_cost
            except Exception as e:
                report.installed.append(
                    InstallationReport(
                        component_id=component_id,
                        destination="",
                        token_cost=0,
                        success=False,
                        error=str(e),
                    )
                )
                report.errors.append(f"Failed to install {component_id}: {e}")

        return report
