"""FastMCP server exposing component retrieval tools."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from git import Repo

from skill_retriever.mcp.rationale import generate_rationale
from skill_retriever.mcp.schemas import (
    ComponentDetail,
    ComponentDetailInput,
    ComponentRecommendation,
    DependencyCheckInput,
    DependencyCheckResult,
    DiscoveredRepo,
    DiscoverReposResult,
    EdgeSuggestionResult,
    FailedRepo,
    FeedbackStatusResult,
    HealStatusResult,
    HealthStatus,
    IngestInput,
    IngestResult,
    InstallInput,
    InstallResult,
    ListTrackedReposResult,
    OutcomeReportResult,
    OutcomeStatsResult,
    PipelineRunResult,
    PipelineStatusResult,
    RegisterRepoInput,
    ReportOutcomeInput,
    ReviewSuggestionInput,
    RunPipelineInput,
    SearchInput,
    SearchResult,
    BackfillSecurityInput,
    BackfillSecurityResult,
    LLMFindingAnalysisResult,
    LLMSecurityScanInput,
    LLMSecurityScanResult,
    SecurityAuditInput,
    SecurityAuditResult,
    SecurityFindingResult,
    SecurityScanInput,
    SecurityScanResult,
    SecurityStatus,
    SyncStatusResult,
    TrackedRepo,
    UnregisterRepoInput,
)
from skill_retriever.entities.components import ComponentMetadata
from skill_retriever.memory.component_memory import ComponentMemory
from skill_retriever.nodes.retrieval.context_assembler import estimate_tokens

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.memory.metadata_store import MetadataStore
    from skill_retriever.memory.vector_store import FAISSVectorStore
    from skill_retriever.workflows.pipeline import RetrievalPipeline

# Configure logging to stderr (CRITICAL: never print to stdout for MCP)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("skill-retriever")

# Global state (lazy initialized)
_pipeline: RetrievalPipeline | None = None
_graph_store: GraphStore | None = None
_vector_store: FAISSVectorStore | None = None
_metadata_store: MetadataStore | None = None
_component_memory: ComponentMemory | None = None
_outcome_tracker: "OutcomeTracker | None" = None
_sync_manager: "SyncManager | None" = None
_init_lock = asyncio.Lock()

if TYPE_CHECKING:
    from skill_retriever.memory.outcome_tracker import OutcomeTracker

# Persistent storage directory (survives restarts)
_STORAGE_DIR = Path.home() / ".skill-retriever" / "data"
_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Paths for persistence
_GRAPH_PATH = _STORAGE_DIR / "graph.json"
_VECTOR_DIR = _STORAGE_DIR / "vectors"
_METADATA_PATH = _STORAGE_DIR / "metadata.json"
_COMPONENT_MEMORY_PATH = _STORAGE_DIR / "component-memory.json"
_OUTCOME_TRACKER_PATH = _STORAGE_DIR / "outcome-tracker.json"
_INGESTION_CACHE_PATH = _STORAGE_DIR / "ingestion-cache.json"

if TYPE_CHECKING:
    from skill_retriever.memory.outcome_tracker import OutcomeTracker
    from skill_retriever.sync.manager import SyncManager


async def get_pipeline() -> RetrievalPipeline:
    """Get or initialize the retrieval pipeline, loading from persistent storage."""
    global _pipeline, _graph_store, _vector_store, _metadata_store, _component_memory, _outcome_tracker

    async with _init_lock:
        if _pipeline is None:
            from skill_retriever.memory.graph_store import NetworkXGraphStore
            from skill_retriever.memory.metadata_store import (
                MetadataStore as MetaStore,
            )
            from skill_retriever.memory.outcome_tracker import (
                OutcomeTracker as OTracker,
            )
            from skill_retriever.memory.vector_store import FAISSVectorStore as VectorStore
            from skill_retriever.workflows.pipeline import RetrievalPipeline

            logger.info("Initializing retrieval pipeline...")

            # Initialize graph store and load from disk if exists
            _graph_store = NetworkXGraphStore()
            if _GRAPH_PATH.exists():
                try:
                    _graph_store.load(str(_GRAPH_PATH))
                    logger.info("Loaded graph store: %d nodes", _graph_store.node_count())
                except Exception:
                    logger.exception("Failed to load graph store, starting fresh")

            # Initialize vector store and load from disk if exists
            _vector_store = VectorStore()
            if _VECTOR_DIR.exists():
                try:
                    _vector_store.load(str(_VECTOR_DIR))
                    logger.info("Loaded vector store: %d vectors", _vector_store.count)
                except Exception:
                    logger.exception("Failed to load vector store, starting fresh")

            # Initialize component memory for usage tracking (LRNG-03)
            _component_memory = ComponentMemory.load(str(_COMPONENT_MEMORY_PATH))

            # Initialize outcome tracker for execution feedback (LRNG-05)
            _outcome_tracker = OTracker.load(str(_OUTCOME_TRACKER_PATH))
            logger.info(
                "Loaded outcome tracker: %d components tracked",
                len(_outcome_tracker.outcomes)
            )

            _pipeline = RetrievalPipeline(
                _graph_store, _vector_store, component_memory=_component_memory
            )

            # Initialize metadata store with persistent path
            _metadata_store = MetaStore(_METADATA_PATH)
            logger.info("Loaded metadata store: %d components", len(_metadata_store))

            logger.info("Pipeline initialized with persistent storage at %s", _STORAGE_DIR)

        return _pipeline


async def get_graph_store() -> GraphStore:
    """Get the graph store, initializing if needed."""
    await get_pipeline()  # Ensures stores are initialized
    assert _graph_store is not None
    return _graph_store


async def get_vector_store() -> FAISSVectorStore:
    """Get the vector store, initializing if needed."""
    await get_pipeline()  # Ensures stores are initialized
    assert _vector_store is not None
    return _vector_store


async def get_metadata_store() -> MetadataStore:
    """Get the metadata store, initializing if needed."""
    await get_pipeline()  # Ensures stores are initialized
    assert _metadata_store is not None
    return _metadata_store


async def get_component_memory() -> ComponentMemory:
    """Get the component memory for usage tracking."""
    await get_pipeline()  # Ensures stores are initialized
    assert _component_memory is not None
    return _component_memory


async def get_outcome_tracker() -> "OutcomeTracker":
    """Get the outcome tracker for execution feedback (LRNG-05)."""
    await get_pipeline()  # Ensures stores are initialized
    assert _outcome_tracker is not None
    return _outcome_tracker


def _compute_health_status(metadata: "ComponentMetadata") -> HealthStatus:
    """Compute health status from component git signals (HLTH-01)."""
    from datetime import UTC, datetime, timedelta

    # Determine status based on last_updated
    status = "unknown"
    last_updated_str = None

    if metadata.last_updated:
        last_updated_str = metadata.last_updated.isoformat()
        now = datetime.now(tz=UTC)
        age = now - metadata.last_updated

        if age < timedelta(days=90):
            status = "active"
        elif age < timedelta(days=365):
            status = "stale"
        else:
            status = "abandoned"

    # Determine commit frequency category
    freq = metadata.commit_frequency_30d
    if freq >= 1.0:  # Daily or more
        freq_str = "high"
    elif freq >= 0.25:  # Weekly
        freq_str = "medium"
    elif freq > 0:
        freq_str = "low"
    else:
        freq_str = "unknown"

    return HealthStatus(
        status=status,
        last_updated=last_updated_str,
        commit_frequency=freq_str,
    )


@mcp.tool
async def search_components(input: SearchInput) -> SearchResult:
    """Search components by task."""
    pipeline = await get_pipeline()
    graph_store = await get_graph_store()
    metadata_store = await get_metadata_store()
    component_memory = await get_component_memory()

    # Convert component_type string to enum if provided
    component_type = None
    if input.component_type:
        from skill_retriever.entities.components import ComponentType

        try:
            component_type = ComponentType(input.component_type)
        except ValueError:
            logger.warning("Invalid component type: %s", input.component_type)

    # Execute retrieval
    result = pipeline.retrieve(
        query=input.query,
        component_type=component_type,
        top_k=input.top_k,
    )

    # Build recommendations with rationale and health status
    recommendations: list[ComponentRecommendation] = []
    for comp in result.context.components:
        node = graph_store.get_node(comp.component_id)
        if node is None:
            continue

        rationale = generate_rationale(comp, result.context, graph_store)
        content = node.label  # Future: use raw_content
        token_cost = estimate_tokens(content)

        # Get health status from metadata (HLTH-01)
        health = None
        security = None
        metadata = metadata_store.get(comp.component_id)
        if metadata:
            health = _compute_health_status(metadata)
            # Get security status (SEC-01)
            security = SecurityStatus(
                risk_level=metadata.security_risk_level,
                risk_score=metadata.security_risk_score,
                findings_count=metadata.security_findings_count,
                has_scripts=metadata.has_scripts,
            )

        # Record recommendation for usage tracking (LRNG-03)
        component_memory.record_recommendation(comp.component_id)

        recommendations.append(
            ComponentRecommendation(
                id=comp.component_id,
                name=node.label,
                type=str(node.component_type),
                score=comp.score,
                rationale=rationale,
                token_cost=token_cost,
                health=health,
                security=security,
            )
        )

    # Persist component memory after recording recommendations
    component_memory.save(str(_COMPONENT_MEMORY_PATH))

    # Format conflicts as strings
    conflict_strs = [
        f"{c.component_a} conflicts with {c.component_b}: {c.reason}"
        for c in result.conflicts
    ]

    return SearchResult(
        components=recommendations,
        total_tokens=result.context.total_tokens,
        conflicts=conflict_strs,
        # RETR-06: Abstraction level awareness
        abstraction_level=result.abstraction_level,
        suggested_types=result.suggested_types or [],
    )


@mcp.tool
async def get_component_detail(input: ComponentDetailInput) -> ComponentDetail:
    """Get full component info."""
    graph_store = await get_graph_store()

    node = graph_store.get_node(input.component_id)
    if node is None:
        return ComponentDetail(
            id=input.component_id,
            name="",
            type="",
            description="Component not found",
            tags=[],
            dependencies=[],
            raw_content="",
            token_cost=0,
        )

    # Get dependencies from edges
    edges = graph_store.get_edges(input.component_id)
    from skill_retriever.entities.graph import EdgeType

    dependencies = [
        edge.target_id
        for edge in edges
        if edge.source_id == input.component_id
        and edge.edge_type == EdgeType.DEPENDS_ON
    ]

    # Future: pull full metadata from component memory
    content = node.label
    token_cost = estimate_tokens(content)

    return ComponentDetail(
        id=node.id,
        name=node.label,
        type=str(node.component_type),
        description=node.label,  # Future: use metadata.description
        tags=[],  # Future: use metadata.tags
        dependencies=dependencies,
        raw_content=content,  # Future: use metadata.raw_content
        token_cost=token_cost,
    )


@mcp.tool
async def install_components(input: InstallInput) -> InstallResult:
    """Install components to .claude/."""
    from skill_retriever.mcp.installer import ComponentInstaller
    from skill_retriever.memory.outcome_tracker import OutcomeType

    graph_store = await get_graph_store()
    metadata_store = await get_metadata_store()
    component_memory = await get_component_memory()
    outcome_tracker = await get_outcome_tracker()

    # Determine target directory
    target_dir = Path(input.target_dir).resolve()

    installer = ComponentInstaller(
        graph_store=graph_store,
        metadata_store=metadata_store,
        target_dir=target_dir,
    )

    report = installer.install(
        component_ids=input.component_ids,
        auto_resolve_deps=True,
    )

    # Record outcomes for feedback loop (LRNG-05)
    for result in report.installed:
        if result.success:
            outcome_tracker.record_outcome(
                result.component_id,
                OutcomeType.INSTALL_SUCCESS,
            )
        else:
            outcome_tracker.record_outcome(
                result.component_id,
                OutcomeType.INSTALL_FAILURE,
                context=result.error or "Unknown error",
            )

    # Persist outcome tracker
    outcome_tracker.save(str(_OUTCOME_TRACKER_PATH))

    # Record selections for co-occurrence tracking (LRNG-03)
    installed_ids = [r.component_id for r in report.installed if r.success]
    if installed_ids:
        component_memory.record_selection(installed_ids)
        component_memory.save(str(_COMPONENT_MEMORY_PATH))

    return InstallResult(
        installed=installed_ids,
        skipped=report.skipped,
        errors=report.errors,
    )


@mcp.tool
async def check_dependencies(input: DependencyCheckInput) -> DependencyCheckResult:
    """Check deps and conflicts."""
    graph_store = await get_graph_store()

    from skill_retriever.workflows.dependency_resolver import (
        detect_conflicts,
        resolve_transitive_dependencies,
    )

    # Resolve transitive dependencies
    all_ids, deps_added = resolve_transitive_dependencies(
        input.component_ids, graph_store
    )

    # Detect conflicts
    conflicts = detect_conflicts(all_ids, graph_store)
    conflict_strs = [
        f"{c.component_a} conflicts with {c.component_b}: {c.reason}"
        for c in conflicts
    ]

    return DependencyCheckResult(
        all_components=sorted(all_ids),
        dependencies_added=deps_added,
        conflicts=conflict_strs,
    )


def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner and repo name from GitHub URL.

    Handles:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - git@github.com:owner/repo.git
    - owner/repo
    """
    # Handle git@ format
    if url.startswith("git@"):
        match = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
        if match:
            return match.group(1), match.group(2)

    # Handle https format
    if "github.com" in url:
        match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
        if match:
            return match.group(1), match.group(2)

    # Handle owner/repo format
    if "/" in url and not url.startswith("http"):
        parts = url.strip("/").split("/")
        if len(parts) == 2:
            return parts[0], parts[1].removesuffix(".git")

    msg = f"Could not parse GitHub URL: {url}"
    raise ValueError(msg)


@mcp.tool
async def ingest_repo(input: IngestInput) -> IngestResult:
    """Index a component repository."""
    from skill_retriever.memory.ingestion_cache import IngestionCache

    graph_store = await get_graph_store()
    vector_store = await get_vector_store()
    metadata_store = await get_metadata_store()

    errors: list[str] = []

    try:
        owner, name = _parse_github_url(input.repo_url)
    except ValueError as e:
        return IngestResult(
            components_found=0,
            components_indexed=0,
            components_skipped=0,
            errors=[str(e)],
        )

    # Initialize ingestion cache for incremental updates (SYNC-03)
    ingestion_cache = IngestionCache(_INGESTION_CACHE_PATH)
    repo_key = f"{owner}/{name}"

    # Track counts for return value
    raw_count = 0
    dedup_count = 0
    indexed_count = 0
    skipped_count = 0

    # Clone to temp directory
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        repo_path = Path(tmpdir) / name
        git_repo: Repo | None = None
        try:
            logger.info("Cloning %s/%s to %s", owner, name, repo_path)
            git_repo = Repo.clone_from(input.repo_url, repo_path)
        except Exception as e:
            return IngestResult(
                components_found=0,
                components_indexed=0,
                components_skipped=0,
                errors=[f"Failed to clone repository: {e}"],
            )

        # Run crawler
        from skill_retriever.nodes.ingestion.crawler import RepositoryCrawler

        crawler = RepositoryCrawler(owner, name, repo_path)
        components = crawler.crawl()

        if not components:
            return IngestResult(
                components_found=0,
                components_indexed=0,
                components_skipped=0,
                errors=["No components found in repository"],
            )

        # Deduplicate components using entity resolution
        from skill_retriever.nodes.ingestion.resolver import EntityResolver

        raw_count = len(components)
        resolver = EntityResolver(fuzzy_threshold=80.0, embedding_threshold=0.85)
        components = resolver.resolve(components)
        dedup_count = raw_count - len(components)
        if dedup_count > 0:
            logger.info("Entity resolution removed %d duplicates", dedup_count)

        # Security scan all components (SEC-01)
        from skill_retriever.security import SecurityScanner

        scanner = SecurityScanner()
        scanned_components: list[ComponentMetadata] = []
        for comp in components:
            scan_result = scanner.scan_component(comp)
            # Create new component with security fields populated
            scanned_comp = ComponentMetadata(
                id=comp.id,
                name=comp.name,
                component_type=comp.component_type,
                description=comp.description,
                tags=list(comp.tags),
                author=comp.author,
                version=comp.version,
                last_updated=comp.last_updated,
                commit_count=comp.commit_count,
                commit_frequency_30d=comp.commit_frequency_30d,
                raw_content=comp.raw_content,
                parameters=dict(comp.parameters),
                dependencies=list(comp.dependencies),
                tools=list(comp.tools),
                source_repo=comp.source_repo,
                source_path=comp.source_path,
                category=comp.category,
                install_url=comp.install_url,
                # Security fields
                security_risk_level=scan_result.risk_level.value,
                security_risk_score=scan_result.risk_score,
                security_findings_count=scan_result.finding_count,
                has_scripts=scan_result.has_scripts,
            )
            scanned_components.append(scanned_comp)
        components = scanned_components
        logger.info("Security scanned %d components", len(components))

        # Add to graph store
        from skill_retriever.entities.graph import GraphEdge, GraphNode

        for comp in components:
            try:
                # Incremental ingestion: skip unchanged components (SYNC-03)
                content_for_hash = f"{comp.name}|{comp.description}|{comp.raw_content}"
                if input.incremental and ingestion_cache.is_unchanged(
                    repo_key, comp.id, content_for_hash
                ):
                    skipped_count += 1
                    continue

                # Add node to graph
                node = GraphNode(
                    id=comp.id,
                    component_type=comp.component_type,
                    label=comp.name,
                    embedding_id=comp.id,
                )
                graph_store.add_node(node)

                # Add dependency edges
                from skill_retriever.entities.graph import EdgeType

                for dep_id in comp.dependencies:
                    edge = GraphEdge(
                        source_id=comp.id,
                        target_id=dep_id,
                        edge_type=EdgeType.DEPENDS_ON,
                    )
                    graph_store.add_edge(edge)

                # Generate and add embedding
                from skill_retriever.nodes.retrieval.vector_search import (
                    _get_embedding_model,  # pyright: ignore[reportPrivateUsage]
                )

                model = _get_embedding_model()
                text = f"{comp.name} {comp.description}"
                embeddings = list(model.embed([text]))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                if embeddings:
                    vector_store.add(comp.id, embeddings[0])  # pyright: ignore[reportUnknownArgumentType]

                # Store component metadata for installation lookup
                metadata_store.add(comp)

                # Update ingestion cache
                ingestion_cache.update_hash(repo_key, comp.id, content_for_hash)

                indexed_count += 1
            except Exception as e:
                errors.append(f"Failed to index {comp.id}: {e}")
                logger.exception("Failed to index component %s", comp.id)

        # Persist all stores after indexing
        metadata_store.save()
        ingestion_cache.save()
        graph_store.save(str(_GRAPH_PATH))
        vector_store.save(str(_VECTOR_DIR))
        logger.info("Persisted stores: graph=%d nodes, vectors=%d",
                    graph_store.node_count(), vector_store.count)

        # Close git repo to release file handles (Windows compatibility)
        if git_repo is not None:
            git_repo.close()

    return IngestResult(
        components_found=raw_count,
        components_indexed=indexed_count,
        components_skipped=skipped_count,
        components_deduplicated=dedup_count,
        errors=errors,
    )


async def get_sync_manager() -> "SyncManager":
    """Get or initialize the sync manager."""
    global _sync_manager

    async with _init_lock:
        if _sync_manager is None:
            from skill_retriever.sync.config import SYNC_CONFIG
            from skill_retriever.sync.manager import SyncManager

            graph_store = await get_graph_store()
            vector_store = await get_vector_store()
            metadata_store = await get_metadata_store()

            _sync_manager = SyncManager(
                config=SYNC_CONFIG,
                graph_store=graph_store,
                vector_store=vector_store,
                metadata_store=metadata_store,
            )
            logger.info("Sync manager initialized")

        return _sync_manager


# ---------------------------------------------------------------------------
# Sync management tools (SYNC-01, SYNC-02)
# ---------------------------------------------------------------------------


@mcp.tool
async def register_repo(input: RegisterRepoInput) -> TrackedRepo:
    """Register a repo for auto-sync."""
    sync_manager = await get_sync_manager()

    try:
        owner, name = _parse_github_url(input.repo_url)
    except ValueError as e:
        raise ValueError(str(e)) from e

    sync_manager.register_repo(
        url=input.repo_url,
        owner=owner,
        name=name,
        webhook_enabled=input.webhook_enabled,
        poll_enabled=input.poll_enabled,
    )

    repo = sync_manager.registry.get(owner, name)
    if not repo:
        raise RuntimeError("Failed to register repo")

    return TrackedRepo(
        owner=repo.owner,
        name=repo.name,
        url=repo.url,
        last_ingested=repo.last_ingested.isoformat() if repo.last_ingested else None,
        last_commit_sha=repo.last_commit_sha,
        webhook_enabled=repo.webhook_enabled,
        poll_enabled=repo.poll_enabled,
    )


@mcp.tool
async def unregister_repo(input: UnregisterRepoInput) -> bool:
    """Unregister a repo from auto-sync."""
    sync_manager = await get_sync_manager()
    return sync_manager.unregister_repo(input.owner, input.name)


@mcp.tool
async def list_tracked_repos() -> ListTrackedReposResult:
    """List all tracked repos."""
    sync_manager = await get_sync_manager()
    repos = sync_manager.registry.list_all()

    tracked = [
        TrackedRepo(
            owner=r.owner,
            name=r.name,
            url=r.url,
            last_ingested=r.last_ingested.isoformat() if r.last_ingested else None,
            last_commit_sha=r.last_commit_sha,
            webhook_enabled=r.webhook_enabled,
            poll_enabled=r.poll_enabled,
        )
        for r in repos
    ]

    return ListTrackedReposResult(repos=tracked, total=len(tracked))


@mcp.tool
async def sync_status() -> SyncStatusResult:
    """Get sync system status."""
    from skill_retriever.sync.config import SYNC_CONFIG

    sync_manager = await get_sync_manager()
    repos = sync_manager.registry.list_all()

    return SyncStatusResult(
        webhook_server_running=sync_manager._webhook._site is not None,
        webhook_port=SYNC_CONFIG.webhook_port,
        polling_enabled=SYNC_CONFIG.poll_enabled,
        poll_interval_seconds=SYNC_CONFIG.poll_interval_seconds,
        tracked_repos=len(repos),
    )


@mcp.tool
async def start_sync_server() -> str:
    """Start webhook server and poller."""
    sync_manager = await get_sync_manager()
    await sync_manager.start()
    return "Sync server started"


@mcp.tool
async def stop_sync_server() -> str:
    """Stop webhook server and poller."""
    sync_manager = await get_sync_manager()
    await sync_manager.stop()
    return "Sync server stopped"


@mcp.tool
async def poll_repos_now() -> str:
    """Trigger immediate poll of all repos."""
    sync_manager = await get_sync_manager()
    await sync_manager.poll_now()
    return "Poll completed"


# ---------------------------------------------------------------------------
# Discovery pipeline tools (OSS-01, HEAL-01)
# ---------------------------------------------------------------------------


@mcp.tool
async def run_discovery_pipeline(input: RunPipelineInput) -> PipelineRunResult:
    """Run the discovery and ingestion pipeline."""
    from skill_retriever.sync.pipeline import DiscoveryPipeline

    pipeline = DiscoveryPipeline(
        min_score=input.min_score,
        max_new_repos=input.max_new_repos,
    )

    result = await pipeline.run(dry_run=input.dry_run)

    return PipelineRunResult(
        discovered_count=result.discovered_count,
        new_repos_count=result.new_repos_count,
        ingested_count=result.ingested_count,
        failed_count=result.failed_count,
        healed_count=result.healed_count,
        skipped_count=result.skipped_count,
        errors=result.errors,
        duration_seconds=result.duration_seconds,
    )


@mcp.tool
async def discover_repos() -> DiscoverReposResult:
    """Discover skill repositories from GitHub."""
    from skill_retriever.sync.oss_scout import OSSScout
    from skill_retriever.sync.registry import RepoRegistry

    scout = OSSScout()
    discovered = scout.discover(force_refresh=True)

    # Get new repos count
    registry = RepoRegistry()
    new_repos = scout.get_new_repos(registry)

    repos = [
        DiscoveredRepo(
            owner=r.owner,
            name=r.name,
            url=r.url,
            stars=r.stars,
            description=r.description,
            updated_at=r.updated_at.isoformat(),
            topics=r.topics,
            score=r.score,
        )
        for r in discovered
    ]

    return DiscoverReposResult(
        repos=repos,
        total=len(repos),
        new_count=len(new_repos),
    )


@mcp.tool
async def get_heal_status() -> HealStatusResult:
    """Get auto-heal status and failures."""
    from skill_retriever.sync.auto_heal import AutoHealer

    healer = AutoHealer()
    status = healer.get_status()
    healable = healer.get_healable_failures()
    healable_keys = {h.repo_key for h in healable}

    failures = [
        FailedRepo(
            repo_key=f.repo_key,
            failure_type=f.failure_type.value,
            error_message=f.error_message,
            first_failure=f.timestamp.isoformat(),
            retry_count=f.retry_count,
            healable=f.repo_key in healable_keys,
        )
        for f in healer.state.failures.values()
    ]

    return HealStatusResult(
        total_failures=status["total_failures"],
        healable_count=status["healable"],
        healed_count=status["total_healed"],
        failures=failures,
    )


@mcp.tool
async def get_pipeline_status() -> PipelineStatusResult:
    """Get discovery pipeline status."""
    from skill_retriever.sync.auto_heal import AutoHealer
    from skill_retriever.sync.pipeline import DiscoveryPipeline

    pipeline = DiscoveryPipeline()
    status = pipeline.get_status()

    healer = AutoHealer()
    heal_status = healer.get_status()
    healable = healer.get_healable_failures()
    healable_keys = {h.repo_key for h in healable}

    failures = [
        FailedRepo(
            repo_key=f.repo_key,
            failure_type=f.failure_type.value,
            error_message=f.error_message,
            first_failure=f.timestamp.isoformat(),
            retry_count=f.retry_count,
            healable=f.repo_key in healable_keys,
        )
        for f in healer.state.failures.values()
    ]

    return PipelineStatusResult(
        scout_cache_path=status["scout_cache"],
        min_score=status["min_score"],
        max_new_repos=status["max_new_repos"],
        heal_status=HealStatusResult(
            total_failures=heal_status["total_failures"],
            healable_count=heal_status["healable"],
            healed_count=heal_status["total_healed"],
            failures=failures,
        ),
    )


@mcp.tool
async def clear_heal_failures() -> str:
    """Clear all tracked failures from auto-heal."""
    from skill_retriever.sync.auto_heal import AutoHealer

    healer = AutoHealer()
    count = len(healer.state.failures)
    healer.state.failures.clear()
    healer.state.total_healed = 0
    healer.state.total_failed = 0
    healer._save_state()

    return f"Cleared {count} failure records"


# ---------------------------------------------------------------------------
# Outcome tracking tools (LRNG-05)
# ---------------------------------------------------------------------------


@mcp.tool
async def report_outcome(input: ReportOutcomeInput) -> str:
    """Report a component outcome (used, removed, deprecated)."""
    from skill_retriever.memory.outcome_tracker import OutcomeType

    outcome_tracker = await get_outcome_tracker()

    # Map string to OutcomeType
    outcome_map = {
        "used": OutcomeType.USED_IN_SESSION,
        "removed": OutcomeType.REMOVED_BY_USER,
        "deprecated": OutcomeType.DEPRECATED,
    }

    outcome_type = outcome_map.get(input.outcome.lower())
    if outcome_type is None:
        return f"Invalid outcome type: {input.outcome}. Use: used, removed, deprecated"

    outcome_tracker.record_outcome(
        input.component_id,
        outcome_type,
        context=input.context,
    )
    outcome_tracker.save(str(_OUTCOME_TRACKER_PATH))

    return f"Recorded {input.outcome} outcome for {input.component_id}"


@mcp.tool
async def get_outcome_stats(component_id: str) -> OutcomeStatsResult:
    """Get outcome statistics for a component."""
    outcome_tracker = await get_outcome_tracker()

    stats = outcome_tracker.outcomes.get(component_id)
    if stats is None:
        return OutcomeStatsResult(
            component_id=component_id,
            install_successes=0,
            install_failures=0,
            usage_count=0,
            removal_count=0,
            success_rate=0.0,
        )

    return OutcomeStatsResult(
        component_id=component_id,
        install_successes=stats.install_successes,
        install_failures=stats.install_failures,
        usage_count=stats.usage_count,
        removal_count=stats.removal_count,
        success_rate=stats.success_rate,
    )


@mcp.tool
async def get_outcome_report() -> OutcomeReportResult:
    """Get overall outcome report with problematic components."""
    outcome_tracker = await get_outcome_tracker()

    return OutcomeReportResult(
        total_components=len(outcome_tracker.outcomes),
        problematic_components=outcome_tracker.get_problematic_components(),
        frequently_removed=outcome_tracker.get_frequently_removed(),
        potential_conflicts=outcome_tracker.get_co_failure_pairs(),
    )


# ---------------------------------------------------------------------------
# Feedback engine tools (LRNG-06)
# ---------------------------------------------------------------------------


@mcp.tool
async def analyze_feedback() -> FeedbackStatusResult:
    """Analyze usage patterns and generate edge suggestions."""
    from skill_retriever.memory.feedback_engine import FeedbackEngine

    component_memory = await get_component_memory()
    outcome_tracker = await get_outcome_tracker()

    engine = FeedbackEngine()
    engine.analyze(component_memory, outcome_tracker)
    status = engine.get_status()

    return FeedbackStatusResult(
        pending_suggestions=status["pending_suggestions"],
        total_suggestions=status["total_suggestions"],
        applied_count=status["applied_count"],
        last_analysis=status["last_analysis"],
    )


@mcp.tool
async def get_feedback_suggestions() -> list[EdgeSuggestionResult]:
    """Get pending edge suggestions from feedback analysis."""
    from skill_retriever.memory.feedback_engine import FeedbackEngine

    engine = FeedbackEngine()
    suggestions = engine.get_pending_suggestions()

    return [
        EdgeSuggestionResult(
            suggestion_type=s.suggestion_type.value,
            source_id=s.source_id,
            target_id=s.target_id,
            confidence=s.confidence,
            evidence=s.evidence,
            reviewed=s.reviewed,
        )
        for s in suggestions
    ]


@mcp.tool
async def review_suggestion(input: ReviewSuggestionInput) -> str:
    """Review a pending edge suggestion (accept or reject)."""
    from skill_retriever.memory.feedback_engine import FeedbackEngine, SuggestionType

    engine = FeedbackEngine()

    # Map string to enum
    type_map = {
        "bundles_with": SuggestionType.BUNDLES_WITH,
        "conflicts_with": SuggestionType.CONFLICTS_WITH,
        "supersedes": SuggestionType.SUPERSEDES,
        "strengthen": SuggestionType.STRENGTHEN,
        "weaken": SuggestionType.WEAKEN,
    }

    suggestion_type = type_map.get(input.suggestion_type.lower())
    if suggestion_type is None:
        return f"Invalid suggestion type: {input.suggestion_type}"

    success = engine.review_suggestion(
        input.source_id,
        input.target_id,
        suggestion_type,
        input.accept,
    )

    if success:
        return f"{'Accepted' if input.accept else 'Rejected'} suggestion: {input.suggestion_type} {input.source_id}->{input.target_id}"
    return "Suggestion not found or already reviewed"


@mcp.tool
async def apply_feedback_suggestions() -> str:
    """Apply all accepted suggestions to the graph."""
    from skill_retriever.memory.feedback_engine import FeedbackEngine

    graph_store = await get_graph_store()
    engine = FeedbackEngine()

    applied_count = engine.apply_accepted_suggestions(graph_store)

    if applied_count > 0:
        # Persist updated graph
        graph_store.save(str(_GRAPH_PATH))

    return f"Applied {applied_count} suggestions to graph"


# ---------------------------------------------------------------------------
# Security scanning tools (SEC-01)
# ---------------------------------------------------------------------------


@mcp.tool
async def security_scan(input: SecurityScanInput) -> SecurityScanResult:
    """Scan a component for security vulnerabilities."""
    from skill_retriever.security import SecurityScanner

    metadata_store = await get_metadata_store()
    component = metadata_store.get(input.component_id)

    if component is None:
        return SecurityScanResult(
            component_id=input.component_id,
            risk_level="unknown",
            risk_score=0.0,
            findings=[],
            has_scripts=False,
            is_safe=False,
        )

    scanner = SecurityScanner()
    result = scanner.scan_component(component)

    # Convert findings to schema format
    findings = [
        SecurityFindingResult(
            pattern_name=f.pattern_name,
            category=f.category,
            risk_level=f.risk_level.value,
            description=f.description,
            matched_text=f.matched_text,
            line_number=f.line_number,
            cwe_id=f.cwe_id,
        )
        for f in result.findings
    ]

    return SecurityScanResult(
        component_id=result.component_id,
        risk_level=result.risk_level.value,
        risk_score=result.risk_score,
        findings=findings,
        has_scripts=result.has_scripts,
        is_safe=result.is_safe,
    )


@mcp.tool
async def security_audit(input: SecurityAuditInput) -> SecurityAuditResult:
    """Audit all indexed components for security vulnerabilities."""
    from skill_retriever.security import RiskLevel

    metadata_store = await get_metadata_store()

    # Map threshold string to enum
    threshold_map = {
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MEDIUM,
        "high": RiskLevel.HIGH,
        "critical": RiskLevel.CRITICAL,
    }
    threshold = threshold_map.get(input.risk_level.lower(), RiskLevel.MEDIUM)

    # Collect stats
    total = 0
    safe = 0
    low = 0
    medium = 0
    high = 0
    critical = 0
    flagged: list[str] = []
    finding_counts: dict[str, int] = {}

    for comp_id, component in metadata_store._cache.items():
        total += 1
        risk_level = component.security_risk_level

        if risk_level == "safe" or risk_level == "unknown":
            safe += 1
        elif risk_level == "low":
            low += 1
            if threshold == RiskLevel.LOW:
                flagged.append(comp_id)
        elif risk_level == "medium":
            medium += 1
            if threshold in (RiskLevel.LOW, RiskLevel.MEDIUM):
                flagged.append(comp_id)
        elif risk_level == "high":
            high += 1
            if threshold in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH):
                flagged.append(comp_id)
        elif risk_level == "critical":
            critical += 1
            flagged.append(comp_id)

    # Note: We don't have individual findings stored in metadata,
    # so top_findings will be empty for now
    # In future: could re-scan flagged components to get findings
    top_findings: list[SecurityFindingResult] = []

    return SecurityAuditResult(
        total_components=total,
        scanned_count=total,
        safe_count=safe,
        low_risk_count=low,
        medium_risk_count=medium,
        high_risk_count=high,
        critical_risk_count=critical,
        flagged_components=flagged[:50],  # Limit to 50
        top_findings=top_findings,
    )


@mcp.tool
async def backfill_security_scans(input: BackfillSecurityInput) -> BackfillSecurityResult:
    """Backfill security scans for existing components.

    Scans all components in the metadata store that don't have security data,
    or all components if force_rescan=True.
    """
    from skill_retriever.security import SecurityScanner

    metadata_store = await get_metadata_store()
    scanner = SecurityScanner()

    total = len(metadata_store)
    scanned = 0
    skipped = 0
    errors: list[str] = []

    # Counters by risk level
    safe = 0
    low = 0
    medium = 0
    high = 0
    critical = 0

    for comp_id, component in list(metadata_store._cache.items()):
        # Skip if already scanned (unless force_rescan)
        if not input.force_rescan and component.security_risk_level != "unknown":
            skipped += 1
            continue

        try:
            # Scan component
            result = scanner.scan_component(component)

            # Create updated component with security fields
            updated = ComponentMetadata(
                id=component.id,
                name=component.name,
                component_type=component.component_type,
                description=component.description,
                tags=list(component.tags),
                author=component.author,
                version=component.version,
                last_updated=component.last_updated,
                commit_count=component.commit_count,
                commit_frequency_30d=component.commit_frequency_30d,
                raw_content=component.raw_content,
                parameters=dict(component.parameters),
                dependencies=list(component.dependencies),
                tools=list(component.tools),
                source_repo=component.source_repo,
                source_path=component.source_path,
                category=component.category,
                install_url=component.install_url,
                # Security fields
                security_risk_level=result.risk_level.value,
                security_risk_score=result.risk_score,
                security_findings_count=result.finding_count,
                has_scripts=result.has_scripts,
            )
            metadata_store.add(updated)
            scanned += 1

            # Count by risk level
            level = result.risk_level.value
            if level == "safe":
                safe += 1
            elif level == "low":
                low += 1
            elif level == "medium":
                medium += 1
            elif level == "high":
                high += 1
            elif level == "critical":
                critical += 1

        except Exception as e:
            errors.append(f"Failed to scan {comp_id}: {e}")
            logger.exception("Failed to scan component %s", comp_id)

    # Persist updated metadata
    metadata_store.save()
    logger.info("Backfill complete: scanned %d, skipped %d, errors %d",
                scanned, skipped, len(errors))

    return BackfillSecurityResult(
        total_components=total,
        scanned_count=scanned,
        skipped_count=skipped,
        safe_count=safe,
        low_risk_count=low,
        medium_risk_count=medium,
        high_risk_count=high,
        critical_risk_count=critical,
        errors=errors[:20],  # Limit errors
    )


@mcp.tool
async def security_scan_llm(input: LLMSecurityScanInput) -> LLMSecurityScanResult:
    """Scan a component with LLM-assisted false positive reduction (SEC-02).

    First runs regex-based scan (SEC-01), then uses Claude to analyze
    each finding and determine if it's a true or false positive.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    from skill_retriever.security import LLMSecurityAnalyzer, SecurityScanner

    metadata_store = await get_metadata_store()
    component = metadata_store.get(input.component_id)

    if component is None:
        return LLMSecurityScanResult(
            component_id=input.component_id,
            llm_available=False,
            original_risk_level="unknown",
            adjusted_risk_level="unknown",
            original_risk_score=0.0,
            adjusted_risk_score=0.0,
            overall_assessment="Component not found",
        )

    # First run regex-based scan
    scanner = SecurityScanner()
    scan_result = scanner.scan_component(component)

    # Then run LLM analysis
    analyzer = LLMSecurityAnalyzer()

    if not analyzer.is_available:
        # Return regex results without LLM analysis
        return LLMSecurityScanResult(
            component_id=input.component_id,
            llm_available=False,
            original_risk_level=scan_result.risk_level.value,
            adjusted_risk_level=scan_result.risk_level.value,
            original_risk_score=scan_result.risk_score,
            adjusted_risk_score=scan_result.risk_score,
            overall_assessment="LLM analysis unavailable (ANTHROPIC_API_KEY not set)",
        )

    # Run LLM analysis
    llm_result = await analyzer.analyze(
        scan_result,
        component.raw_content,
        component.name,
        component.description,
    )

    if llm_result is None:
        return LLMSecurityScanResult(
            component_id=input.component_id,
            llm_available=True,
            original_risk_level=scan_result.risk_level.value,
            adjusted_risk_level=scan_result.risk_level.value,
            original_risk_score=scan_result.risk_score,
            adjusted_risk_score=scan_result.risk_score,
            overall_assessment="LLM analysis failed",
        )

    # Convert finding analyses to result format
    finding_analyses = [
        LLMFindingAnalysisResult(
            pattern_name=fa.pattern_name,
            verdict=fa.verdict.value,
            confidence=fa.confidence,
            reasoning=fa.reasoning,
            is_in_documentation=fa.is_in_documentation,
            mitigations=fa.mitigations,
        )
        for fa in llm_result.finding_analyses
    ]

    return LLMSecurityScanResult(
        component_id=input.component_id,
        llm_available=True,
        original_risk_level=llm_result.original_risk_level,
        adjusted_risk_level=llm_result.adjusted_risk_level,
        original_risk_score=llm_result.original_risk_score,
        adjusted_risk_score=llm_result.adjusted_risk_score,
        finding_analyses=finding_analyses,
        overall_assessment=llm_result.overall_assessment,
        false_positive_count=llm_result.false_positive_count,
        true_positive_count=llm_result.true_positive_count,
        context_dependent_count=llm_result.context_dependent_count,
    )


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
