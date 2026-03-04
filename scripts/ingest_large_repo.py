"""Ingest a large repo directly, bypassing MCP timeout constraints.

Usage:
    cd ~/repos/skill-retriever
    uv run python scripts/ingest_large_repo.py https://github.com/openclaw/skills
"""

import sys
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_large_repo.py <repo_url>")
        sys.exit(1)

    repo_url = sys.argv[1]

    # Parse owner/name from URL
    parts = repo_url.rstrip("/").split("/")
    owner, name = parts[-2], parts[-1]
    logger.info("Ingesting %s/%s from %s", owner, name, repo_url)

    # Import after path setup — use same modules as pipeline.py and manager.py
    from git import Repo
    from skill_retriever.nodes.ingestion.crawler import RepositoryCrawler
    from skill_retriever.nodes.ingestion.resolver import EntityResolver
    from skill_retriever.entities.graph import GraphNode, GraphEdge, EdgeType
    from skill_retriever.memory.graph_store import NetworkXGraphStore
    from skill_retriever.memory.vector_store import FAISSVectorStore
    from skill_retriever.memory.metadata_store import MetadataStore
    from skill_retriever.memory.ingestion_cache import IngestionCache
    from skill_retriever.nodes.retrieval.vector_search import _get_embedding_model

    # Paths
    data_dir = Path.home() / ".skill-retriever" / "data"
    graph_path = data_dir / "graph.json"
    vectors_dir = data_dir / "vectors"
    metadata_path = data_dir / "metadata.json"
    cache_path = data_dir / "ingestion-cache.json"

    # Initialize stores (same pattern as pipeline._ingest_repo)
    graph_store = NetworkXGraphStore()
    vector_store = FAISSVectorStore()
    metadata_store = MetadataStore(metadata_path)
    ingestion_cache = IngestionCache(cache_path)
    repo_key = f"{owner}/{name}"

    # Load existing data
    if graph_path.exists():
        graph_store.load(str(graph_path))
        logger.info("Loaded existing graph: %d nodes, %d edges", graph_store.node_count(), graph_store.edge_count())
    if vectors_dir.exists():
        vector_store.load(str(vectors_dir))
        logger.info("Loaded existing vectors")

    # Clone (shallow)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        repo_path = Path(tmpdir) / name
        logger.info("Shallow cloning to %s ...", repo_path)
        Repo.clone_from(repo_url, repo_path, depth=1)
        logger.info("Clone complete.")

        # Crawl
        logger.info("Crawling for components...")
        crawler = RepositoryCrawler(owner, name, repo_path)
        components = crawler.crawl()
        logger.info("Found %d raw components", len(components))

        if not components:
            logger.error("No components found!")
            sys.exit(1)

        # Deduplicate
        resolver = EntityResolver(fuzzy_threshold=80.0, embedding_threshold=0.85)
        before = len(components)
        components = resolver.resolve(components)
        logger.info("After dedup: %d (removed %d)", len(components), before - len(components))

        # Index
        model = _get_embedding_model()
        indexed = 0
        skipped = 0
        errors = 0
        for i, comp in enumerate(components):
            if i % 500 == 0:
                logger.info("Indexing: %d/%d (indexed=%d, skipped=%d)", i, len(components), indexed, skipped)
            try:
                content_for_hash = f"{comp.name}|{comp.description}|{comp.raw_content}"
                if ingestion_cache.is_unchanged(repo_key, comp.id, content_for_hash):
                    skipped += 1
                    continue

                node = GraphNode(
                    id=comp.id,
                    component_type=comp.component_type,
                    label=comp.name,
                    embedding_id=comp.id,
                )
                graph_store.add_node(node)

                for dep_id in comp.dependencies:
                    edge = GraphEdge(
                        source_id=comp.id,
                        target_id=dep_id,
                        edge_type=EdgeType.DEPENDS_ON,
                    )
                    graph_store.add_edge(edge)

                text = f"{comp.name} {comp.description}"
                embeddings = list(model.embed([text]))
                if embeddings:
                    vector_store.add(comp.id, embeddings[0])

                metadata_store.add(comp)
                ingestion_cache.update_hash(repo_key, comp.id, content_for_hash)
                indexed += 1
            except Exception as e:
                errors += 1
                if errors <= 10:
                    logger.error("Failed to index %s: %s", comp.id, e)

        # Save
        logger.info("Persisting stores...")
        metadata_store.save()
        graph_store.save(str(graph_path))
        vector_store.save(str(vectors_dir))
        ingestion_cache.save()

        logger.info("DONE: indexed=%d, skipped=%d, errors=%d, total=%d", indexed, skipped, errors, len(components))


if __name__ == "__main__":
    main()
