"""Automated discovery and ingestion pipeline.

Combines OSS Scout, auto-heal, and ingestion into a single automated flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from skill_retriever.sync.auto_heal import AutoHealer, FailureType
from skill_retriever.sync.oss_scout import OSSScout
from skill_retriever.sync.registry import RepoRegistry

if TYPE_CHECKING:
    from skill_retriever.sync.oss_scout import DiscoveredRepo

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    discovered_count: int
    new_repos_count: int
    ingested_count: int
    failed_count: int
    healed_count: int
    skipped_count: int
    errors: list[str]
    duration_seconds: float


class DiscoveryPipeline:
    """Automated pipeline for discovering and ingesting skill repositories."""

    def __init__(
        self,
        registry_path: Path | None = None,
        github_token: str | None = None,
        min_score: float = 30.0,
        max_new_repos: int = 10,
    ) -> None:
        """Initialize the pipeline.

        Args:
            registry_path: Path to repo registry (default: ~/.skill-retriever/repo-registry.json)
            github_token: GitHub API token for higher rate limits.
            min_score: Minimum quality score to consider a repo (default: 30).
            max_new_repos: Maximum new repos to ingest per run (default: 10).
        """
        self.registry_path = registry_path or (
            Path.home() / ".skill-retriever" / "repo-registry.json"
        )
        self.scout = OSSScout(github_token=github_token)
        self.healer = AutoHealer()
        self.min_score = min_score
        self.max_new_repos = max_new_repos

    async def run(self, dry_run: bool = False) -> PipelineResult:
        """Run the full discovery and ingestion pipeline.

        Args:
            dry_run: If True, discover but don't ingest.

        Returns:
            PipelineResult with statistics.
        """
        start_time = datetime.now()
        errors: list[str] = []
        ingested = 0
        failed = 0
        healed = 0
        skipped = 0

        # Load registry
        registry = RepoRegistry(self.registry_path)

        # 1. Discover repositories
        logger.info("Phase 1: Discovering repositories...")
        discovered = self.scout.discover(force_refresh=True)
        new_repos = self.scout.get_new_repos(registry)

        # Filter by minimum score
        qualified = [r for r in new_repos if r.score >= self.min_score]
        logger.info(
            "Found %d new repos (%d qualified with score >= %.1f)",
            len(new_repos),
            len(qualified),
            self.min_score,
        )

        if dry_run:
            logger.info("Dry run - skipping ingestion")
            for repo in qualified[:self.max_new_repos]:
                logger.info("  Would ingest: %s (score: %.1f)", repo.full_name, repo.score)

            return PipelineResult(
                discovered_count=len(discovered),
                new_repos_count=len(new_repos),
                ingested_count=0,
                failed_count=0,
                healed_count=0,
                skipped_count=len(qualified),
                errors=[],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

        # 2. Ingest new repositories
        logger.info("Phase 2: Ingesting new repositories...")
        for repo in qualified[:self.max_new_repos]:
            try:
                result = await self._ingest_repo(repo, registry)
                if result["indexed"] > 0:
                    ingested += 1
                    logger.info(
                        "Ingested %s: %d components",
                        repo.full_name,
                        result["indexed"],
                    )
                elif result.get("error"):
                    failed += 1
                    self.healer.record_failure(repo.full_name, result["error"])
                    errors.append(f"{repo.full_name}: {result['error']}")
                else:
                    # No components but no error - might be a meta-list
                    skipped += 1
                    logger.info("Skipped %s: no components found", repo.full_name)
            except Exception as e:
                failed += 1
                self.healer.record_failure(repo.full_name, str(e))
                errors.append(f"{repo.full_name}: {e}")
                logger.error("Failed to ingest %s: %s", repo.full_name, e)

        # 3. Attempt to heal previous failures
        logger.info("Phase 3: Attempting to heal previous failures...")
        healable = self.healer.get_healable_failures()
        for failure in healable[:5]:  # Limit healing attempts
            if failure.failure_type == FailureType.NO_COMPONENTS:
                # These often can't be healed without code changes
                continue

            if failure.failure_type in (
                FailureType.NETWORK_ERROR,
                FailureType.RATE_LIMITED,
            ):
                # Retry these
                try:
                    owner, name = failure.repo_key.split("/")
                    from skill_retriever.sync.oss_scout import DiscoveredRepo

                    repo = DiscoveredRepo(
                        owner=owner,
                        name=name,
                        url=f"https://github.com/{failure.repo_key}",
                        stars=0,
                        description="",
                        updated_at=datetime.now(),
                        topics=[],
                        score=0,
                    )
                    result = await self._ingest_repo(repo, registry)
                    if result["indexed"] > 0:
                        self.healer.mark_healed(
                            failure.repo_key,
                            f"Retry successful: {result['indexed']} components",
                        )
                        healed += 1
                except Exception as e:
                    logger.debug("Healing failed for %s: %s", failure.repo_key, e)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        result = PipelineResult(
            discovered_count=len(discovered),
            new_repos_count=len(new_repos),
            ingested_count=ingested,
            failed_count=failed,
            healed_count=healed,
            skipped_count=skipped,
            errors=errors,
            duration_seconds=duration,
        )

        # Save pipeline result
        self._save_run_result(result)

        logger.info(
            "Pipeline complete: %d discovered, %d new, %d ingested, %d failed, %d healed (%.1fs)",
            result.discovered_count,
            result.new_repos_count,
            result.ingested_count,
            result.failed_count,
            result.healed_count,
            result.duration_seconds,
        )

        return result

    async def _ingest_repo(
        self, repo: "DiscoveredRepo", registry: RepoRegistry
    ) -> dict:
        """Ingest a single repository.

        Args:
            repo: The discovered repository to ingest.
            registry: The repo registry.

        Returns:
            Dict with ingestion results.
        """
        # Register the repo first
        registry.register(
            url=repo.url,
            owner=repo.owner,
            name=repo.name,
            poll_enabled=True,
        )

        # Import here to avoid circular imports
        import tempfile
        from pathlib import Path

        from git import Repo

        from skill_retriever.memory.graph_store import NetworkXGraphStore
        from skill_retriever.memory.ingestion_cache import IngestionCache
        from skill_retriever.memory.metadata_store import MetadataStore
        from skill_retriever.memory.vector_store import FAISSVectorStore
        from skill_retriever.nodes.ingestion.crawler import RepositoryCrawler
        from skill_retriever.nodes.ingestion.resolver import EntityResolver
        from skill_retriever.entities.graph import GraphNode, GraphEdge, EdgeType
        from skill_retriever.nodes.retrieval.vector_search import _get_embedding_model

        # Storage paths
        storage_dir = Path.home() / ".skill-retriever" / "data"
        graph_path = storage_dir / "graph.json"
        vector_dir = storage_dir / "vectors"
        metadata_path = storage_dir / "metadata.json"
        cache_path = storage_dir / "ingestion-cache.json"

        # Load stores
        graph_store = NetworkXGraphStore()
        vector_store = FAISSVectorStore()
        metadata_store = MetadataStore(metadata_path)
        ingestion_cache = IngestionCache(cache_path)

        if graph_path.exists():
            graph_store.load(str(graph_path))
        if vector_dir.exists():
            vector_store.load(str(vector_dir))

        model = _get_embedding_model()

        indexed_count = 0
        skipped_count = 0
        errors = []

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            repo_path = Path(tmpdir) / repo.name
            git_repo = None

            try:
                git_repo = Repo.clone_from(repo.url, repo_path)
            except Exception as e:
                return {"error": f"Clone failed: {e}", "indexed": 0}

            # Crawl
            crawler = RepositoryCrawler(repo.owner, repo.name, repo_path)
            components = crawler.crawl()

            if not components:
                if git_repo:
                    git_repo.close()
                return {"error": "No components found", "indexed": 0}

            # Deduplicate
            resolver = EntityResolver(fuzzy_threshold=80.0, embedding_threshold=0.85)
            components = resolver.resolve(components)

            # Index each component
            for comp in components:
                try:
                    content_hash = f"{comp.name}|{comp.description}|{comp.raw_content}"
                    if ingestion_cache.is_unchanged(repo.full_name, comp.id, content_hash):
                        skipped_count += 1
                        continue

                    # Graph node
                    node = GraphNode(
                        id=comp.id,
                        component_type=comp.component_type,
                        label=comp.name,
                        embedding_id=comp.id,
                    )
                    graph_store.add_node(node)

                    # Dependency edges
                    for dep_id in comp.dependencies:
                        edge = GraphEdge(
                            source_id=comp.id,
                            target_id=dep_id,
                            edge_type=EdgeType.DEPENDS_ON,
                        )
                        graph_store.add_edge(edge)

                    # Embedding
                    text = f"{comp.name} {comp.description}"
                    embeddings = list(model.embed([text]))
                    if embeddings:
                        vector_store.add(comp.id, embeddings[0])

                    # Metadata
                    metadata_store.add(comp)

                    # Cache
                    ingestion_cache.update_hash(repo.full_name, comp.id, content_hash)
                    indexed_count += 1

                except Exception as e:
                    errors.append(f"{comp.id}: {e}")

            if git_repo:
                git_repo.close()

        # Save stores
        metadata_store.save()
        ingestion_cache.save()
        graph_store.save(str(graph_path))
        vector_store.save(str(vector_dir))

        return {
            "indexed": indexed_count,
            "skipped": skipped_count,
            "errors": errors[:3] if errors else [],
        }

    def _save_run_result(self, result: PipelineResult) -> None:
        """Save pipeline run result for tracking."""
        history_path = Path.home() / ".skill-retriever" / "pipeline-history.json"

        history = []
        if history_path.exists():
            try:
                with open(history_path, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass

        # Keep last 100 runs
        history = history[-99:]
        history.append({
            "timestamp": datetime.now().isoformat(),
            "discovered": result.discovered_count,
            "new_repos": result.new_repos_count,
            "ingested": result.ingested_count,
            "failed": result.failed_count,
            "healed": result.healed_count,
            "duration": result.duration_seconds,
        })

        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def get_status(self) -> dict:
        """Get pipeline status including healing stats."""
        return {
            "healing": self.healer.get_status(),
            "scout_cache": str(self.scout.cache_path),
            "min_score": self.min_score,
            "max_new_repos": self.max_new_repos,
        }


async def run_pipeline(
    dry_run: bool = False,
    min_score: float = 30.0,
    max_new_repos: int = 10,
) -> PipelineResult:
    """Convenience function to run the discovery pipeline.

    Args:
        dry_run: If True, discover but don't ingest.
        min_score: Minimum quality score for repos.
        max_new_repos: Max new repos per run.

    Returns:
        PipelineResult with statistics.
    """
    pipeline = DiscoveryPipeline(
        min_score=min_score,
        max_new_repos=max_new_repos,
    )
    return await pipeline.run(dry_run=dry_run)
