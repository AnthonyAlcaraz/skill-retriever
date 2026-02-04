"""Polling-based sync for repositories without webhooks."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import httpx

if TYPE_CHECKING:
    from skill_retriever.sync.config import SyncConfig
    from skill_retriever.sync.registry import RepoRegistry

logger = logging.getLogger(__name__)

# Configure logging to stderr
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RepoPoller:
    """Polls GitHub API to detect repository changes.

    For repos without webhooks, periodically checks the latest commit
    SHA and triggers re-ingestion when changes are detected.
    """

    def __init__(
        self,
        config: SyncConfig,
        registry: RepoRegistry,
        on_change: Callable[[str, str, str | None], Coroutine[Any, Any, None]] | None = None,
        github_token: str | None = None,
    ) -> None:
        """Initialize poller.

        Args:
            config: Sync configuration with polling settings.
            registry: Repository registry for tracking state.
            on_change: Async callback when changes detected (owner, name, commit_sha).
            github_token: Optional GitHub token for API rate limits.
        """
        self._config = config
        self._registry = registry
        self._on_change = on_change
        self._github_token = github_token
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def _get_latest_commit(self, owner: str, name: str) -> str | None:
        """Fetch latest commit SHA from GitHub API."""
        url = f"https://api.github.com/repos/{owner}/{name}/commits"
        headers = {"Accept": "application/vnd.github+json"}

        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params={"per_page": 1})
                response.raise_for_status()
                commits = response.json()
                if commits:
                    return str(commits[0]["sha"])
        except httpx.HTTPStatusError as e:
            logger.warning("GitHub API error for %s/%s: %s", owner, name, e)
        except Exception:
            logger.exception("Failed to fetch commits for %s/%s", owner, name)

        return None

    async def _poll_once(self) -> None:
        """Poll all registered repos once."""
        repos = self._registry.list_for_polling()
        if not repos:
            logger.debug("No repos to poll")
            return

        logger.info("Polling %d repos for changes", len(repos))

        for repo in repos:
            try:
                latest_sha = await self._get_latest_commit(repo.owner, repo.name)
                if not latest_sha:
                    continue

                # Check if changed
                if repo.last_commit_sha and repo.last_commit_sha == latest_sha:
                    logger.debug("%s/%s: no changes (SHA: %s)", repo.owner, repo.name, latest_sha[:8])
                    continue

                logger.info(
                    "%s/%s: change detected (old: %s, new: %s)",
                    repo.owner,
                    repo.name,
                    repo.last_commit_sha[:8] if repo.last_commit_sha else "none",
                    latest_sha[:8],
                )

                # Trigger callback
                if self._on_change:
                    await self._on_change(repo.owner, repo.name, latest_sha)

            except Exception:
                logger.exception("Failed to poll %s/%s", repo.owner, repo.name)

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            await self._poll_once()
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Poller started (interval: %d seconds)", self._config.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Poller stopped")

    async def poll_now(self) -> None:
        """Trigger an immediate poll (for testing or manual refresh)."""
        await self._poll_once()
