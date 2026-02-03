"""Frontmatter parsing and normalization for component files."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import Any

import frontmatter

logger = logging.getLogger(__name__)


def parse_component_file(file_path: Path) -> tuple[dict[str, Any], str]:
    """Parse a markdown file with YAML frontmatter.

    Returns a tuple of (metadata_dict, body_content).
    If the file has no frontmatter or parsing fails, metadata_dict will be empty.
    """
    try:
        post = frontmatter.load(str(file_path))
    except FileNotFoundError:
        logger.warning("File not found: %s", file_path)
        return {}, ""
    except Exception as e:
        # Catch YAML parsing errors (malformed frontmatter)
        logger.warning("Failed to parse frontmatter in %s: %s", file_path, e)
        # Fall back to reading raw content without frontmatter
        try:
            content = file_path.read_text(encoding="utf-8")
            return {}, content
        except Exception:
            return {}, ""

    return dict(post.metadata), post.content


def normalize_frontmatter(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw frontmatter into a consistent schema.

    - Maps ``allowed-tools`` and ``allowed_tools`` to ``tools``.
    - Maps ``requires``, ``depends``, ``depends-on``, ``depends_on`` to ``dependencies``.
    - Ensures ``tags`` is always a list (splits on commas if string).
    - Ensures ``tools`` is always a list.
    - Ensures ``dependencies`` is always a list.
    - Strips whitespace from ``name`` and ``description``.
    """
    result = dict(raw)

    # Merge allowed-tools / allowed_tools into tools
    for key in ("allowed-tools", "allowed_tools"):
        if key in result:
            result.setdefault("tools", result.pop(key))

    # Merge dependency aliases into dependencies
    for key in ("requires", "depends", "depends-on", "depends_on"):
        if key in result:
            result.setdefault("dependencies", result.pop(key))

    # Ensure tools is a list
    tools = result.get("tools")
    if tools is None:
        result["tools"] = []
    elif isinstance(tools, str):
        result["tools"] = [t.strip() for t in tools.split(",") if t.strip()]

    # Ensure tags is a list
    tags = result.get("tags")
    if tags is None:
        result["tags"] = []
    elif isinstance(tags, str):
        result["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    # Ensure dependencies is a list
    deps = result.get("dependencies")
    if deps is None:
        result["dependencies"] = []
    elif isinstance(deps, str):
        result["dependencies"] = [d.strip() for d in deps.split(",") if d.strip()]

    # Strip whitespace from name and description
    if "name" in result and isinstance(result["name"], str):
        result["name"] = result["name"].strip()
    if "description" in result and isinstance(result["description"], str):
        result["description"] = result["description"].strip()

    # Ensure version is a string (YAML may parse "2.0" as float)
    if "version" in result and not isinstance(result["version"], str):
        result["version"] = str(result["version"])

    return result
