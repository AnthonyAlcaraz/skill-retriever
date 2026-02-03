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


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class ComponentRecommendation(BaseModel):
    """A recommended component with score and rationale."""

    id: str
    name: str
    type: str
    score: float
    rationale: str
    token_cost: int


class SearchResult(BaseModel):
    """Result of component search."""

    components: list[ComponentRecommendation]
    total_tokens: int
    conflicts: list[str]


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
    errors: list[str]
