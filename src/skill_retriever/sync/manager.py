"""Sync manager that orchestrates webhook and polling-based sync."""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from git import Repo

if TYPE_CHECKING:
    from skill_retriever.memory.graph_store import GraphStore
    from skill_retriever.memory.metadata_store import MetadataStore
    from skill_retriever.memory.vector_store import FAISSVectorStore
    from skill_retriever.sync.config import SyncConfig

logger = logging.getLogger(__name__)

# Configure logging to stderr
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SyncManager:
    """Manages auto-sync for tracked repositories.

    Orchestrates:
    - Webhook server for real-time push event handling
    - Polling for repos without webhooks
    - Re-ingestion when changes are detected
    """

    def __init__(
        self,
        config: SyncConfig,
        graph_store: GraphStore,
        vector_store: FAISSVectorStore,
        metadata_store: MetadataStore,
        github_token: str | None = None,
    ) -> None:
        """Initialize sync manager.

        Args:
            config: Sync configuration.
            graph_store: Graph store for indexing.
            vector_store: Vector store for embeddings.
            metadata_store: Metadata store for component info.
            github_token: Optional GitHub token for polling API.
        """
        from skill_retriever.sync.poller import RepoPoller
        from skill_retriever.sync.registry import RepoRegistry
        from skill_retriever.sync.webhook import WebhookServer

        self._config = config
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._metadata_store = metadata_store

        # Initialize registry
        self._registry = RepoRegistry(config.registry_path)

        # Initialize webhook server
        self._webhook = WebhookServer(config, on_push=self._handle_change)

        # Initialize poller
        self._poller = RepoPoller(
            config,
            self._registry,
            on_change=self._handle_change,
            github_token=github_token,
        )

        self._ingestion_lock = asyncio.Lock()

    @property
    def registry(self) -> RepoRegistry:
        """Get the repository registry."""
        from skill_retriever.sync.registry import RepoRegistry

        return self._registry

    async def _handle_change(self, owner: str, name: str, commit_sha: str | None) -> None:
        """Handle detected repo change by re-ingesting."""
        # Check if repo is registered
        repo = self._registry.get(owner, name)
        if not repo:
            logger.warning("Received change event for unregistered repo %s/%s", owner, name)
            return

        # Prevent concurrent ingestion of same repo
        async with self._ingestion_lock:
            logger.info("Re-ingesting %s/%s (commit: %s)", owner, name, commit_sha or "unknown")

            try:
                await self._ingest_repo(owner, name, repo.url)
                self._registry.mark_ingested(owner, name, commit_sha)
                logger.info("Successfully re-ingested %s/%s", owner, name)
            except Exception:
                logger.exception("Failed to re-ingest %s/%s", owner, name)

    async def _ingest_repo(self, owner: str, name: str, url: str) -> int:
        """Clone and ingest a repository.

        Returns number of components indexed.
        """
        from skill_retriever.entities.graph import EdgeType, GraphEdge, GraphNode
        from skill_retriever.memory.ingestion_cache import IngestionCache
        from skill_retriever.nodes.ingestion.crawler import RepositoryCrawler
        from skill_retriever.nodes.ingestion.resolver import EntityResolver
        from skill_retriever.nodes.retrieval.vector_search import _get_embedding_model

        # Initialize ingestion cache for incremental updates
        cache_path = Path(tempfile.gettempdir()) / "skill-retriever-ingestion-cache.json"
        ingestion_cache = IngestionCache(cache_path)
        repo_key = f"{owner}/{name}"

        indexed_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / name

            # Clone
            logger.info("Cloning %s to %s", url, repo_path)
            Repo.clone_from(url, repo_path)

            # Crawl
            crawler = RepositoryCrawler(owner, name, repo_path)
            components = crawler.crawl()

            if not components:
                logger.warning("No components found in %s/%s", owner, name)
                return 0

            # Deduplicate
            resolver = EntityResolver(fuzzy_threshold=80.0, embedding_threshold=0.85)
            components = resolver.resolve(components)

            # Get embedding model
            model = _get_embedding_model()

            # Index components
            for comp in components:
                # Incremental: skip unchanged
                content_for_hash = f"{comp.name}|{comp.description}|{comp.raw_content}"
                if self._config.incremental and ingestion_cache.is_unchanged(
                    repo_key, comp.id, content_for_hash
                ):
                    continue

                # Add node
                node = GraphNode(
                    id=comp.id,
                    component_type=comp.component_type,
                    label=comp.name,
                    embedding_id=comp.id,
                )
                self._graph_store.add_node(node)

                # Add dependency edges
                for dep_id in comp.dependencies:
                    edge = GraphEdge(
                        source_id=comp.id,
                        target_id=dep_id,
                        edge_type=EdgeType.DEPENDS_ON,
                    )
                    self._graph_store.add_edge(edge)

                # Generate and add embedding
                text = f"{comp.name} {comp.description}"
                embeddings = list(model.embed([text]))
                if embeddings:
                    self._vector_store.add(comp.id, embeddings[0])

                # Store metadata
                self._metadata_store.add(comp)

                # Update cache
                ingestion_cache.update_hash(repo_key, comp.id, content_for_hash)
                indexed_count += 1

            # Persist
            self._metadata_store.save()
            ingestion_cache.save()

        return indexed_count

    async def start(self) -> None:
        """Start sync services (webhook server and poller)."""
        # Start webhook server
        await self._webhook.start()

        # Start poller if enabled
        if self._config.poll_enabled:
            await self._poller.start()

        logger.info("Sync manager started")

    async def stop(self) -> None:
        """Stop sync services."""
        await self._poller.stop()
        await self._webhook.stop()
        logger.info("Sync manager stopped")

    def register_repo(
        self,
        url: str,
        owner: str,
        name: str,
        webhook_enabled: bool = False,
        poll_enabled: bool = True,
    ) -> None:
        """Register a repository for sync tracking."""
        self._registry.register(url, owner, name, webhook_enabled, poll_enabled)

    def unregister_repo(self, owner: str, name: str) -> bool:
        """Unregister a repository from sync tracking."""
        return self._registry.unregister(owner, name)

    async def ingest_now(self, owner: str, name: str) -> int:
        """Manually trigger ingestion for a registered repo."""
        repo = self._registry.get(owner, name)
        if not repo:
            raise ValueError(f"Repo {owner}/{name} not registered")

        async with self._ingestion_lock:
            count = await self._ingest_repo(owner, name, repo.url)
            self._registry.mark_ingested(owner, name)
            return count

    async def poll_now(self) -> None:
        """Manually trigger a poll cycle."""
        await self._poller.poll_now()
