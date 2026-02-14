"""Repository crawler that applies extraction strategies to discover components."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003

from skill_retriever.entities import ComponentMetadata  # noqa: TC001
from skill_retriever.nodes.ingestion.extractors import (
    AwesomeListStrategy,
    Davila7Strategy,
    FlatDirectoryStrategy,
    GenericMarkdownStrategy,
    PackageJsonStrategy,
    PluginMarketplaceStrategy,
    PythonModuleStrategy,
    ReadmeFallbackStrategy,
)
from skill_retriever.nodes.ingestion.git_signals import extract_git_signals

logger = logging.getLogger(__name__)


class RepositoryCrawler:
    """Crawl a local repository clone and extract component metadata.

    Uses multiple strategies to extract components from different file types.
    Markdown strategies are tried in priority order (first match wins).
    Python strategy runs independently to capture source code components.
    """

    def __init__(self, repo_owner: str, repo_name: str, repo_path: Path) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_path = repo_path
        # Markdown strategies (first match wins - more specific first)
        self.markdown_strategies = [
            Davila7Strategy(repo_owner, repo_name),
            PluginMarketplaceStrategy(repo_owner, repo_name),  # plugins/{name}/skills/
            FlatDirectoryStrategy(repo_owner, repo_name),       # .claude/{type}/
            GenericMarkdownStrategy(repo_owner, repo_name),     # Any markdown with name frontmatter
        ]
        # Special strategies that run independently
        self.python_strategy = PythonModuleStrategy(repo_owner, repo_name)
        self.awesome_list_strategy = AwesomeListStrategy(repo_owner, repo_name)
        self.package_json_strategy = PackageJsonStrategy(repo_owner, repo_name)
        self.readme_fallback_strategy = ReadmeFallbackStrategy(repo_owner, repo_name)

    def crawl(self) -> list[ComponentMetadata]:
        """Discover and extract all components from the repository."""
        components: list[ComponentMetadata] = []

        # Find markdown strategy
        md_strategy = None
        for s in self.markdown_strategies:
            if s.can_handle(self.repo_path):
                md_strategy = s
                logger.info(
                    "Using %s for markdown in %s/%s",
                    type(s).__name__,
                    self.repo_owner,
                    self.repo_name,
                )
                break

        # Extract markdown components
        if md_strategy is not None:
            md_files = md_strategy.discover(self.repo_path)
            logger.info("Discovered %d markdown component files", len(md_files))
            for file_path in md_files:
                comp = self._extract_with_signals(md_strategy, file_path)
                if comp:
                    components.append(comp)

        # Extract Python components (independent of markdown)
        if self.python_strategy.can_handle(self.repo_path):
            logger.info(
                "Using PythonModuleStrategy for Python in %s/%s",
                self.repo_owner,
                self.repo_name,
            )
            py_files = self.python_strategy.discover(self.repo_path)
            logger.info("Discovered %d Python component files", len(py_files))
            for file_path in py_files:
                comp = self._extract_with_signals(self.python_strategy, file_path)
                if comp:
                    components.append(comp)

        # Try awesome list strategy for curated repos (extracts from README)
        if self.awesome_list_strategy.can_handle(self.repo_path):
            logger.info(
                "Using AwesomeListStrategy for %s/%s",
                self.repo_owner,
                self.repo_name,
            )
            awesome_components = self.awesome_list_strategy.extract_all(self.repo_path)
            logger.info("Extracted %d components from awesome list", len(awesome_components))
            components.extend(awesome_components)

        # Extract package.json components (independent of markdown)
        if self.package_json_strategy.can_handle(self.repo_path):
            logger.info(
                "Using PackageJsonStrategy for %s/%s",
                self.repo_owner,
                self.repo_name,
            )
            pkg_files = self.package_json_strategy.discover(self.repo_path)
            logger.info("Discovered %d package.json files", len(pkg_files))
            for file_path in pkg_files:
                comp = self._extract_with_signals(self.package_json_strategy, file_path)
                if comp:
                    components.append(comp)

        # Fallback: if still no components, try README-based extraction
        if not components and self.readme_fallback_strategy.can_handle(self.repo_path):
            logger.info(
                "Using ReadmeFallbackStrategy (catch-all) for %s/%s",
                self.repo_owner,
                self.repo_name,
            )
            readme_files = self.readme_fallback_strategy.discover(self.repo_path)
            for file_path in readme_files:
                comp = self._extract_with_signals(self.readme_fallback_strategy, file_path)
                if comp:
                    components.append(comp)

        if not components:
            logger.warning("No components found for %s", self.repo_path)

        logger.info("Extracted %d total components", len(components))
        return components

    def _extract_with_signals(
        self, strategy: object, file_path: Path
    ) -> ComponentMetadata | None:
        """Extract component and merge git signals."""
        try:
            # Strategy has extract method - use Any to avoid type issues
            component: ComponentMetadata | None = strategy.extract(file_path, self.repo_path)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("Failed to extract %s: %s", file_path, e)
            return None

        if component is None:
            return None

        # Merge git signals
        try:
            rel_path = str(file_path.relative_to(self.repo_path))
            signals = extract_git_signals(self.repo_path, rel_path)
            update_fields: dict[str, object] = {}
            if signals.get("last_updated") is not None:
                update_fields["last_updated"] = signals["last_updated"]
            if signals.get("commit_count", 0) > 0:
                update_fields["commit_count"] = signals["commit_count"]
            if signals.get("commit_frequency_30d", 0.0) > 0.0:
                update_fields["commit_frequency_30d"] = signals["commit_frequency_30d"]
            if update_fields:
                component = component.model_copy(update=update_fields)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        except Exception:
            logger.debug("Git signals failed for %s", file_path, exc_info=True)

        return component  # pyright: ignore[reportUnknownVariableType]
