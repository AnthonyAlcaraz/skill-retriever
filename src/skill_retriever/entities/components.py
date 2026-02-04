"""Domain models for Claude Code components."""

from __future__ import annotations

import re
from datetime import datetime  # noqa: TC003
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComponentType(StrEnum):
    """The 7 component types found in Claude Code ecosystem."""

    AGENT = "agent"
    SKILL = "skill"
    COMMAND = "command"
    SETTING = "setting"
    MCP = "mcp"
    HOOK = "hook"
    SANDBOX = "sandbox"


class ComponentMetadata(BaseModel):
    """Immutable metadata describing a single Claude Code component.

    Serves as the canonical data contract for crawlers, memory stores,
    and retrieval queries.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        description=(
            "Deterministic ID: {repo_owner}/{repo_name}/{type}/{normalized_name}. "
            "Name portion is lowercased with spaces replaced by hyphens."
        )
    )
    name: str
    component_type: ComponentType
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    version: str = ""
    last_updated: datetime | None = None
    commit_count: int = 0
    commit_frequency_30d: float = 0.0
    raw_content: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    source_repo: str = ""
    source_path: str = ""
    category: str = ""
    install_url: str | None = None  # For curated list entries that link to external repos

    # Security fields (SEC-01)
    security_risk_level: str = "unknown"  # safe, low, medium, high, critical
    security_risk_score: float = 0.0  # 0-100
    security_findings_count: int = 0
    has_scripts: bool = False

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, v: str) -> str:
        """Normalize the name portion of the ID to lowercase-hyphenated."""
        parts = v.strip().split("/")
        if len(parts) >= 4:
            # Normalize only the name portion (everything after the 3rd slash)
            name_part = "/".join(parts[3:])
            normalized = re.sub(r"\s+", "-", name_part.strip()).lower()
            return f"{parts[0]}/{parts[1]}/{parts[2]}/{normalized}"
        return v.strip().lower()

    @classmethod
    def generate_id(
        cls,
        repo_owner: str,
        repo_name: str,
        component_type: ComponentType,
        name: str,
    ) -> str:
        """Produce a deterministic ID string from source location and name."""
        normalized_name = re.sub(r"\s+", "-", name.strip()).lower()
        return f"{repo_owner}/{repo_name}/{component_type.value}/{normalized_name}"
