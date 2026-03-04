"""Lightweight validation gate for components before indexing.

Checks for common quality issues like missing descriptions,
stub content, placeholder patterns, and missing tool definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentMetadata


@dataclass
class ValidationResult:
    """Result of component validation."""

    passed: bool
    issues: list[str] = field(default_factory=list)


def validate_component(comp: ComponentMetadata) -> ValidationResult:
    """Lightweight validation before indexing.

    Checks for:
    - Name length and presence
    - Description length and presence
    - Stub content (<50 chars)
    - Placeholder patterns (TODO, FIXME, etc.) in short content
    - Agent components without tools defined

    Args:
        comp: Component metadata to validate.

    Returns:
        ValidationResult with passed status and any issues found.
    """
    issues: list[str] = []

    if not comp.name or len(comp.name) < 2:
        issues.append("Name too short or missing")
    if not comp.description or len(comp.description) < 10:
        issues.append("Description too short or missing")

    if comp.raw_content and len(comp.raw_content.strip()) < 50:
        issues.append("Content appears to be a stub (<50 chars)")

    placeholder_patterns = ["TODO", "FIXME", "placeholder", "template only"]
    if comp.raw_content:
        content_lower = comp.raw_content.lower()
        for p in placeholder_patterns:
            if p.lower() in content_lower and len(comp.raw_content) < 200:
                issues.append(f"Possible placeholder content: contains '{p}'")
                break

    if comp.component_type.value == "agent" and not comp.tools:
        issues.append("Agent has no tools defined")

    return ValidationResult(passed=len(issues) == 0, issues=issues)
