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
    components_deduplicated: int = Field(default=0, description="Duplicates removed by entity resolution")
    errors: list[str]


# ---------------------------------------------------------------------------
# Sync management models (SYNC-01, SYNC-02)
# ---------------------------------------------------------------------------


class RegisterRepoInput(BaseModel):
    """Input for register_repo tool."""

    repo_url: str = Field(description="GitHub repository URL")
    webhook_enabled: bool = Field(default=False, description="Webhook configured for this repo")
    poll_enabled: bool = Field(default=True, description="Enable polling for changes")


class UnregisterRepoInput(BaseModel):
    """Input for unregister_repo tool."""

    owner: str = Field(description="Repository owner")
    name: str = Field(description="Repository name")


class TrackedRepo(BaseModel):
    """A tracked repository."""

    owner: str
    name: str
    url: str
    last_ingested: str | None = Field(default=None, description="ISO timestamp of last ingestion")
    last_commit_sha: str | None = Field(default=None, description="SHA of last ingested commit")
    webhook_enabled: bool
    poll_enabled: bool


class ListTrackedReposResult(BaseModel):
    """Result of list_tracked_repos tool."""

    repos: list[TrackedRepo]
    total: int


class SyncStatusResult(BaseModel):
    """Result of sync_status tool."""

    webhook_server_running: bool
    webhook_port: int
    polling_enabled: bool
    poll_interval_seconds: int
    tracked_repos: int


# ---------------------------------------------------------------------------
# Discovery pipeline models (OSS-01, HEAL-01)
# ---------------------------------------------------------------------------


class RunPipelineInput(BaseModel):
    """Input for run_discovery_pipeline tool."""

    dry_run: bool = Field(default=False, description="Discover but don't ingest")
    min_score: float = Field(default=30.0, description="Minimum quality score")
    max_new_repos: int = Field(default=10, description="Max repos to ingest")


class PipelineRunResult(BaseModel):
    """Result of discovery pipeline run."""

    discovered_count: int = Field(description="Total repos discovered")
    new_repos_count: int = Field(description="Repos not yet tracked")
    ingested_count: int = Field(description="Successfully ingested")
    failed_count: int = Field(description="Failed to ingest")
    healed_count: int = Field(description="Previous failures healed")
    skipped_count: int = Field(description="Skipped (dry-run or meta-list)")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    duration_seconds: float = Field(description="Pipeline run duration")


class DiscoveredRepo(BaseModel):
    """A discovered repository from OSS Scout."""

    owner: str
    name: str
    url: str
    stars: int
    description: str
    updated_at: str = Field(description="ISO timestamp of last update")
    topics: list[str]
    score: float = Field(description="Quality score (0-100)")


class DiscoverReposResult(BaseModel):
    """Result of discover_repos tool."""

    repos: list[DiscoveredRepo]
    total: int
    new_count: int = Field(description="Repos not yet tracked")


class FailedRepo(BaseModel):
    """A failed repository from auto-heal tracking."""

    repo_key: str = Field(description="owner/name format")
    failure_type: str = Field(description="Type of failure")
    error_message: str
    first_failure: str = Field(description="ISO timestamp")
    retry_count: int
    healable: bool = Field(description="Can be auto-healed")


class HealStatusResult(BaseModel):
    """Result of get_heal_status tool."""

    total_failures: int
    healable_count: int
    healed_count: int
    failures: list[FailedRepo]


class PipelineStatusResult(BaseModel):
    """Result of get_pipeline_status tool."""

    scout_cache_path: str
    min_score: float
    max_new_repos: int
    heal_status: HealStatusResult


# ---------------------------------------------------------------------------
# Outcome tracking models (LRNG-05)
# ---------------------------------------------------------------------------


class ReportOutcomeInput(BaseModel):
    """Input for report_outcome tool."""

    component_id: str = Field(description="Component ID")
    outcome: str = Field(description="Outcome type: used, removed, deprecated")
    context: str = Field(default="", description="Optional context")


class OutcomeStatsResult(BaseModel):
    """Outcome statistics for a component."""

    component_id: str
    install_successes: int
    install_failures: int
    usage_count: int
    removal_count: int
    success_rate: float


class OutcomeReportResult(BaseModel):
    """Result of get_outcome_report tool."""

    total_components: int
    problematic_components: list[str] = Field(description="Components with low success rates")
    frequently_removed: list[str] = Field(description="Components often removed by users")
    potential_conflicts: list[tuple[str, str, int]] = Field(
        description="(component_a, component_b, co_failure_count) tuples"
    )


# ---------------------------------------------------------------------------
# Feedback engine models (LRNG-06)
# ---------------------------------------------------------------------------


class EdgeSuggestionResult(BaseModel):
    """An edge suggestion from the feedback engine."""

    suggestion_type: str = Field(description="bundles_with, conflicts_with, etc.")
    source_id: str
    target_id: str
    confidence: float
    evidence: str
    reviewed: bool


class FeedbackStatusResult(BaseModel):
    """Result of get_feedback_status tool."""

    pending_suggestions: int
    total_suggestions: int
    applied_count: int
    last_analysis: str | None


class ReviewSuggestionInput(BaseModel):
    """Input for review_suggestion tool."""

    source_id: str = Field(description="Source component ID")
    target_id: str = Field(description="Target component ID")
    suggestion_type: str = Field(description="bundles_with or conflicts_with")
    accept: bool = Field(description="Whether to accept the suggestion")
