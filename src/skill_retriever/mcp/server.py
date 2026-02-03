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
    IngestInput,
    IngestResult,
    InstallInput,
    InstallResult,
    SearchInput,
    SearchResult,
)
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
_init_lock = asyncio.Lock()


async def get_pipeline() -> RetrievalPipeline:
    """Get or initialize the retrieval pipeline."""
    global _pipeline, _graph_store, _vector_store, _metadata_store

    async with _init_lock:
        if _pipeline is None:
            from pathlib import Path

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


@mcp.tool
async def search_components(input: SearchInput) -> SearchResult:
    """Search components by task."""
    pipeline = await get_pipeline()
    graph_store = await get_graph_store()

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

    # Build recommendations with rationale
    recommendations: list[ComponentRecommendation] = []
    for comp in result.context.components:
        node = graph_store.get_node(comp.component_id)
        if node is None:
            continue

        rationale = generate_rationale(comp, result.context, graph_store)
        content = node.label  # Future: use raw_content
        token_cost = estimate_tokens(content)

        recommendations.append(
            ComponentRecommendation(
                id=comp.component_id,
                name=node.label,
                type=str(node.component_type),
                score=comp.score,
                rationale=rationale,
                token_cost=token_cost,
            )
        )

    # Format conflicts as strings
    conflict_strs = [
        f"{c.component_a} conflicts with {c.component_b}: {c.reason}"
        for c in result.conflicts
    ]

    return SearchResult(
        components=recommendations,
        total_tokens=result.context.total_tokens,
        conflicts=conflict_strs,
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
    from pathlib import Path

    from skill_retriever.mcp.installer import ComponentInstaller

    graph_store = await get_graph_store()
    metadata_store = await get_metadata_store()

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

    return InstallResult(
        installed=[r.component_id for r in report.installed if r.success],
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
            errors=[str(e)],
        )

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
                errors=["No components found in repository"],
            )

        # Add to graph store
        from skill_retriever.entities.graph import GraphEdge, GraphNode

        indexed_count = 0
        for comp in components:
            try:
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

                indexed_count += 1
            except Exception as e:
                errors.append(f"Failed to index {comp.id}: {e}")
                logger.exception("Failed to index component %s", comp.id)

        # Persist metadata store after all components indexed
        metadata_store.save()

    return IngestResult(
        components_found=len(components),
        components_indexed=indexed_count,
        errors=errors,
    )


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
