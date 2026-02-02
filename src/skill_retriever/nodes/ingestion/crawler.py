"""Repository crawler that applies extraction strategies to discover components."""

from __future__ import annotations

import logging
from pathlib import Path

from skill_retriever.entities import ComponentMetadata
from skill_retriever.nodes.ingestion.extractors import (
    Davila7Strategy,
    FlatDirectoryStrategy,
    GenericMarkdownStrategy,
)
from skill_retriever.nodes.ingestion.git_signals import extract_git_signals

logger = logging.getLogger(__name__)


class RepositoryCrawler:
    """Crawl a local repository clone and extract component metadata.

    Strategies are tried in order; the first one whose ``can_handle``
    returns True is used for the entire repository.
    """

    def __init__(self, repo_owner: str, repo_name: str, repo_path: Path) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.strategies = [
            Davila7Strategy(repo_owner, repo_name),
            FlatDirectoryStrategy(repo_owner, repo_name),
            GenericMarkdownStrategy(repo_owner, repo_name),
        ]

    def crawl(self) -> list[ComponentMetadata]:
        """Discover and extract all components from the repository."""
        strategy = None
        for s in self.strategies:
            if s.can_handle(self.repo_path):
                strategy = s
                logger.info(
                    "Using %s for %s/%s",
                    type(s).__name__,
                    self.repo_owner,
                    self.repo_name,
                )
                break

        if strategy is None:
            logger.warning("No strategy matched for %s", self.repo_path)
            return []

        files = strategy.discover(self.repo_path)
        logger.info("Discovered %d component files", len(files))

        components: list[ComponentMetadata] = []
        for file_path in files:
            component = strategy.extract(file_path, self.repo_path)
            if component is None:
                continue

            # Merge git signals
            try:
                rel_path = str(file_path.relative_to(self.repo_path))
                signals = extract_git_signals(self.repo_path, rel_path)
                # Filter out None values and zero defaults that shouldn't override
                update_fields: dict[str, object] = {}
                if signals.get("last_updated") is not None:
                    update_fields["last_updated"] = signals["last_updated"]
                if signals.get("commit_count", 0) > 0:
                    update_fields["commit_count"] = signals["commit_count"]
                if signals.get("commit_frequency_30d", 0.0) > 0.0:
                    update_fields["commit_frequency_30d"] = signals["commit_frequency_30d"]
                if update_fields:
                    component = component.model_copy(update=update_fields)
            except Exception:
                logger.debug("Git signals failed for %s", file_path, exc_info=True)

            components.append(component)

        logger.info("Extracted %d components", len(components))
        return components
