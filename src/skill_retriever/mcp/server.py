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
    HealthStatus,
    IngestInput,
    IngestResult,
    InstallInput,
    InstallResult,
    ListTrackedReposResult,
    RegisterRepoInput,
    SearchInput,
    SearchResult,
    SyncStatusResult,
    TrackedRepo,
    UnregisterRepoInput,
)
from skill_retriever.memory.component_memory import ComponentMemory
from skill_retriever.nodes.retrieval.context_assembler import estimate_tokens

if TYPE_CHECKING:
    from skill_retriever.entities.components import ComponentMetadata
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
_sync_manager: "SyncManager | None" = None
_init_lock = asyncio.Lock()

# Path for component memory persistence
_COMPONENT_MEMORY_PATH = Path(tempfile.gettempdir()) / "skill-retriever-memory.json"

if TYPE_CHECKING:
    from skill_retriever.sync.manager import SyncManager


async def get_pipeline() -> RetrievalPipeline:
    """Get or initialize the retrieval pipeline."""
    global _pipeline, _graph_store, _vector_store, _metadata_store, _component_memory

    async with _init_lock:
        if _pipeline is None:
            from skill_retriever.memory.graph_store import NetworkXGraphStore
            from skill_retriever.memory.metadata_store import (
                MetadataStore as MetaStore,
            )
            from skill_retriever.memory.vector_store import FAISSVectorStore as VectorStore
            from skill_retriever.workflows.pipeline import RetrievalPipeline

            logger.info("Initializing retrieval pipeline...")
            _graph_store = NetworkXGraphStore()
            _vector_store = VectorStore()
            _pipeline = RetrievalPipeline(_graph_store, _vector_store)

            # Initialize metadata store in temp location (future: configurable)
            metadata_path = Path(tempfile.gettempdir()) / "skill-retriever-metadata.json"
            _metadata_store = MetaStore(metadata_path)

            # Initialize component memory for usage tracking (LRNG-03)
            _component_memory = ComponentMemory.load(str(_COMPONENT_MEMORY_PATH))
            logger.info("Pipeline initialized")

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
        metadata = metadata_store.get(comp.component_id)
        if metadata:
            health = _compute_health_status(metadata)

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

    graph_store = await get_graph_store()
    metadata_store = await get_metadata_store()
    component_memory = await get_component_memory()

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
    cache_path = Path(tempfile.gettempdir()) / "skill-retriever-ingestion-cache.json"
    ingestion_cache = IngestionCache(cache_path)
    repo_key = f"{owner}/{name}"

    # Track counts for return value
    raw_count = 0
    dedup_count = 0
    indexed_count = 0
    skipped_count = 0

    # Clone to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / name
        try:
            logger.info("Cloning %s/%s to %s", owner, name, repo_path)
            Repo.clone_from(input.repo_url, repo_path)
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

        # Persist stores after all components indexed
        metadata_store.save()
        ingestion_cache.save()

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


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
