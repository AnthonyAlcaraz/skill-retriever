"""Pydantic input/output models for MCP tool handlers."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Input models (keep descriptions under 10 words)
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
    """Input for search_components tool."""

    query: str = Field(description="Task description")
    top_k: int = Field(default=5, description="Max results")
    component_type: str | None = Field(default=None, description="Filter by type")


class ComponentDetailInput(BaseModel):
    """Input for get_component_detail tool."""

    component_id: str = Field(description="Component ID")


class InstallInput(BaseModel):
    """Input for install_components tool."""

    component_ids: list[str] = Field(description="Component IDs to install")
    target_dir: str = Field(default=".", description="Installation directory")


class DependencyCheckInput(BaseModel):
    """Input for check_dependencies tool."""

    component_ids: list[str] = Field(description="Component IDs to check")


class IngestInput(BaseModel):
    """Input for ingest_repo tool."""

    repo_url: str = Field(description="GitHub repository URL")
    incremental: bool = Field(default=True, description="Skip unchanged files")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    """Component health indicators based on git signals."""

    status: str = Field(description="active, stale, or abandoned")
    last_updated: str | None = Field(default=None, description="ISO date of last commit")
    commit_frequency: str = Field(default="unknown", description="high, medium, low, or unknown")


class ComponentRecommendation(BaseModel):
    """A recommended component with score and rationale."""

    id: str
    name: str
    type: str
    score: float
    rationale: str
    token_cost: int
    health: HealthStatus | None = Field(default=None, description="Component health status")


class SearchResult(BaseModel):
    """Result of component search."""

    components: list[ComponentRecommendation]
    total_tokens: int
    conflicts: list[str]
    # RETR-06: Abstraction level awareness
    abstraction_level: str = Field(default="medium", description="high, medium, or low")
    suggested_types: list[str] = Field(default_factory=list, description="Suggested component types")


class ComponentDetail(BaseModel):
    """Full component information."""

    id: str
    name: str
    type: str
    description: str
    tags: list[str]
    dependencies: list[str]
    raw_content: str
    token_cost: int


class InstallResult(BaseModel):
    """Result of component installation."""

    installed: list[str]
    skipped: list[str]
    errors: list[str]


class DependencyCheckResult(BaseModel):
    """Result of dependency check."""

    all_components: list[str]
    dependencies_added: list[str]
    conflicts: list[str]


class IngestResult(BaseModel):
    """Result of repository ingestion."""

    components_found: int
    components_indexed: int
    components_skipped: int = Field(default=0, description="Unchanged components skipped")
    errors: list[str]
