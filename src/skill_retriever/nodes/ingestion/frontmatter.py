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


def _flatten_to_str_list(value: Any) -> list[str]:
    """Flatten a value of unknown type into a flat list of strings.

    Handles: None, str (comma-split), list (recurse), dict (flatten keys+values),
    and other scalars (str-cast).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, dict):
        items: list[str] = []
        for k, v in value.items():
            if isinstance(v, list):
                items.extend(str(i) for i in v)
            elif isinstance(v, str):
                items.append(v)
            elif v is not None:
                items.append(str(v))
            # Also include the key as context (e.g. "mcp" from {mcp: ['rube']})
            items.append(str(k))
        return [s.strip() for s in items if s.strip()]
    if isinstance(value, list):
        flat: list[str] = []
        for item in value:
            if isinstance(item, str):
                flat.append(item.strip())
            elif isinstance(item, dict):
                flat.extend(_flatten_to_str_list(item))
            elif item is not None:
                flat.append(str(item))
        return [s for s in flat if s]
    return [str(value)]


def normalize_frontmatter(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw frontmatter into a consistent schema.

    - Maps ``allowed-tools`` and ``allowed_tools`` to ``tools``.
    - Maps ``requires``, ``depends``, ``depends-on``, ``depends_on`` to ``dependencies``.
    - Ensures ``tags`` is always a list (splits on commas if string).
    - Ensures ``tools`` is always a list of strings.
    - Ensures ``dependencies`` is always a list of strings.
    - Handles dict values like ``{mcp: ['rube']}`` by flattening to ``['rube', 'mcp']``.
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

    # Ensure tools is a flat list of strings
    result["tools"] = _flatten_to_str_list(result.get("tools"))

    # Ensure tags is a flat list of strings
    result["tags"] = _flatten_to_str_list(result.get("tags"))

    # Ensure dependencies is a flat list of strings
    result["dependencies"] = _flatten_to_str_list(result.get("dependencies"))

    # Strip whitespace from name and description
    if "name" in result and isinstance(result["name"], str):
        result["name"] = result["name"].strip()
    if "description" in result and isinstance(result["description"], str):
        result["description"] = result["description"].strip()

    # Ensure version is a string (YAML may parse "2.0" as float)
    if "version" in result and not isinstance(result["version"], str):
        result["version"] = str(result["version"])

    return result
