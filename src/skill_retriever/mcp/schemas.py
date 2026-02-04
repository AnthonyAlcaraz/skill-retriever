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


class SecurityStatus(BaseModel):
    """Component security indicators from vulnerability scanning (SEC-01)."""

    risk_level: str = Field(default="unknown", description="safe, low, medium, high, or critical")
    risk_score: float = Field(default=0.0, description="0-100 risk score")
    findings_count: int = Field(default=0, description="Number of security findings")
    has_scripts: bool = Field(default=False, description="Contains executable scripts")


class ComponentRecommendation(BaseModel):
    """A recommended component with score and rationale."""

    id: str
    name: str
    type: str
    score: float
    rationale: str
    token_cost: int
    health: HealthStatus | None = Field(default=None, description="Component health status")
    security: SecurityStatus | None = Field(default=None, description="Security scan results")


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


# ---------------------------------------------------------------------------
# Security scanning models (SEC-01)
# ---------------------------------------------------------------------------


class SecurityScanInput(BaseModel):
    """Input for security_scan tool."""

    component_id: str = Field(description="Component ID to scan")


class SecurityFindingResult(BaseModel):
    """A security finding from scanning."""

    pattern_name: str = Field(description="Name of the vulnerability pattern")
    category: str = Field(description="exfiltration, credential_access, privilege_escalation, obfuscation")
    risk_level: str = Field(description="critical, high, medium, low")
    description: str = Field(description="What this finding means")
    matched_text: str = Field(default="", description="The matched code snippet (truncated)")
    line_number: int | None = Field(default=None, description="Line number in content")
    cwe_id: str | None = Field(default=None, description="Common Weakness Enumeration ID")


class SecurityScanResult(BaseModel):
    """Result of security scan."""

    component_id: str
    risk_level: str = Field(description="safe, low, medium, high, or critical")
    risk_score: float = Field(description="0-100 risk score")
    findings: list[SecurityFindingResult] = Field(default_factory=list)
    has_scripts: bool = Field(default=False, description="Contains executable scripts")
    is_safe: bool = Field(description="True if no significant vulnerabilities found")


class SecurityAuditInput(BaseModel):
    """Input for security_audit tool."""

    risk_level: str = Field(default="medium", description="Minimum risk level to report: low, medium, high, critical")


class SecurityAuditResult(BaseModel):
    """Result of security audit across all components."""

    total_components: int
    scanned_count: int
    safe_count: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    critical_risk_count: int
    flagged_components: list[str] = Field(description="Component IDs at or above threshold")
    top_findings: list[SecurityFindingResult] = Field(description="Most common finding patterns")


class BackfillSecurityInput(BaseModel):
    """Input for backfill_security_scans tool."""

    batch_size: int = Field(default=100, description="Components to scan per batch")
    force_rescan: bool = Field(default=False, description="Rescan already-scanned components")


class BackfillSecurityResult(BaseModel):
    """Result of security backfill operation."""

    total_components: int
    scanned_count: int
    skipped_count: int = Field(description="Already scanned (not force_rescan)")
    safe_count: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    critical_risk_count: int
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM Security Analysis models (SEC-02)
# ---------------------------------------------------------------------------


class LLMSecurityScanInput(BaseModel):
    """Input for security_scan_llm tool."""

    component_id: str = Field(description="Component ID to analyze")


class LLMFindingAnalysisResult(BaseModel):
    """LLM analysis of a single finding."""

    pattern_name: str
    verdict: str = Field(description="true_positive, false_positive, context_dependent, needs_review")
    confidence: float = Field(description="0.0-1.0 confidence in verdict")
    reasoning: str = Field(description="Explanation of the verdict")
    is_in_documentation: bool = Field(description="Pattern is in docs/examples, not executable")
    mitigations: list[str] = Field(default_factory=list, description="Suggested fixes if true positive")


class LLMSecurityScanResult(BaseModel):
    """Result of LLM-assisted security scan."""

    component_id: str
    llm_available: bool = Field(description="Whether LLM analysis was performed")
    original_risk_level: str
    adjusted_risk_level: str
    original_risk_score: float
    adjusted_risk_score: float
    finding_analyses: list[LLMFindingAnalysisResult] = Field(default_factory=list)
    overall_assessment: str = Field(default="")
    false_positive_count: int = 0
    true_positive_count: int = 0
    context_dependent_count: int = 0
