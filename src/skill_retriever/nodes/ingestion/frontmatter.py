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
    If the file has no frontmatter, metadata_dict will be empty.
    """
    try:
        post = frontmatter.load(str(file_path))
    except FileNotFoundError:
        logger.warning("File not found: %s", file_path)
        return {}, ""

    return dict(post.metadata), post.content


def normalize_frontmatter(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw frontmatter into a consistent schema.

    - Maps ``allowed-tools`` and ``allowed_tools`` to ``tools``.
    - Ensures ``tags`` is always a list (splits on commas if string).
    - Ensures ``tools`` is always a list.
    - Strips whitespace from ``name`` and ``description``.
    """
    result = dict(raw)

    # Merge allowed-tools / allowed_tools into tools
    for key in ("allowed-tools", "allowed_tools"):
        if key in result:
            result.setdefault("tools", result.pop(key))

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

    # Strip whitespace from name and description
    if "name" in result and isinstance(result["name"], str):
        result["name"] = result["name"].strip()
    if "description" in result and isinstance(result["description"], str):
        result["description"] = result["description"].strip()

    return result
