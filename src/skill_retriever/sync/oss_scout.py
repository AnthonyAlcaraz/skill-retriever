"""OSS Scout - Automated discovery of Claude Code skill repositories.

Searches GitHub for high-quality skill repositories and automatically
ingests them into the skill-retriever index.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from skill_retriever.sync.registry import RepoRegistry

logger = logging.getLogger(__name__)

# Search queries to find skill repositories
SEARCH_QUERIES = [
    "claude code skills",
    "claude skills agent",
    "awesome claude skills",
    "claude code agents",
    "anthropic skills",
    "claude mcp server",
    "claude code commands",
    "agent skills specification",
]

# Minimum criteria for a quality repository
MIN_STARS = 5
MIN_RECENT_ACTIVITY_DAYS = 180  # Updated within last 6 months

# Known high-quality repos to always include (even if search misses them)
SEED_REPOS = [
    "anthropics/skills",
    "obra/superpowers",
    "wshobson/agents",
    "VoltAgent/awesome-agent-skills",
    "BehiSecc/awesome-claude-skills",
    "ComposioHQ/awesome-claude-skills",
    "travisvn/awesome-claude-skills",
    "daymade/claude-code-skills",
    "zxkane/aws-skills",
]


@dataclass
class DiscoveredRepo:
    """A discovered repository candidate."""

    owner: str
    name: str
    url: str
    stars: int
    description: str
    updated_at: datetime
    topics: list[str]
    score: float = 0.0  # Quality score

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class OSSScout:
    """Discovers and evaluates Claude Code skill repositories on GitHub."""

    def __init__(
        self,
        github_token: str | None = None,
        cache_path: Path | None = None,
    ) -> None:
        """Initialize the OSS Scout.

        Args:
            github_token: Optional GitHub API token for higher rate limits.
            cache_path: Path to cache discovered repos (default: ~/.skill-retriever/scout-cache.json)
        """
        self.github_token = github_token or self._load_github_token()
        self.cache_path = cache_path or (
            Path.home() / ".skill-retriever" / "scout-cache.json"
        )
        self._discovered: dict[str, DiscoveredRepo] = {}
        self._load_cache()

    def _load_github_token(self) -> str | None:
        """Try to load GitHub token from common locations."""
        # Try environment variable
        import os

        token = os.environ.get("GITHUB_TOKEN")
        if token:
            return token

        # Try gh CLI config
        gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
        if gh_config.exists():
            try:
                import yaml

                with open(gh_config) as f:
                    config = yaml.safe_load(f)
                    if config and "github.com" in config:
                        return config["github.com"].get("oauth_token")
            except Exception:
                pass

        # Try token file
        token_file = Path.home() / ".config" / "github-token.txt"
        if token_file.exists():
            return token_file.read_text().strip()

        return None

    def _load_cache(self) -> None:
        """Load discovered repos from cache."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                for repo_data in data.get("repos", []):
                    repo = DiscoveredRepo(
                        owner=repo_data["owner"],
                        name=repo_data["name"],
                        url=repo_data["url"],
                        stars=repo_data["stars"],
                        description=repo_data.get("description", ""),
                        updated_at=datetime.fromisoformat(repo_data["updated_at"]),
                        topics=repo_data.get("topics", []),
                        score=repo_data.get("score", 0.0),
                    )
                    self._discovered[repo.full_name] = repo
                logger.info("Loaded %d repos from scout cache", len(self._discovered))
            except Exception as e:
                logger.warning("Failed to load scout cache: %s", e)

    def _save_cache(self) -> None:
        """Save discovered repos to cache."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_scan": datetime.now().isoformat(),
            "repos": [
                {
                    "owner": repo.owner,
                    "name": repo.name,
                    "url": repo.url,
                    "stars": repo.stars,
                    "description": repo.description,
                    "updated_at": repo.updated_at.isoformat(),
                    "topics": repo.topics,
                    "score": repo.score,
                }
                for repo in self._discovered.values()
            ],
        }
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d repos to scout cache", len(self._discovered))

    def _github_search(self, query: str) -> list[dict]:
        """Search GitHub for repositories matching query."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        results = []
        try:
            with httpx.Client(timeout=30.0) as client:
                # Search code repositories
                url = "https://api.github.com/search/repositories"
                params = {
                    "q": f"{query} in:name,description,readme",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 30,
                }
                response = client.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results.extend(data.get("items", []))
                elif response.status_code == 403:
                    logger.warning("GitHub rate limit reached")
                else:
                    logger.warning(
                        "GitHub search failed: %s %s",
                        response.status_code,
                        response.text[:100],
                    )
        except Exception as e:
            logger.warning("GitHub search error: %s", e)

        return results

    def _evaluate_repo(self, repo_data: dict) -> DiscoveredRepo | None:
        """Evaluate if a repository is a valid skill repository."""
        try:
            updated_at = datetime.fromisoformat(
                repo_data["updated_at"].replace("Z", "+00:00")
            )
            # Remove timezone info for comparison
            updated_at = updated_at.replace(tzinfo=None)

            repo = DiscoveredRepo(
                owner=repo_data["owner"]["login"],
                name=repo_data["name"],
                url=repo_data["html_url"],
                stars=repo_data["stargazers_count"],
                description=repo_data.get("description") or "",
                updated_at=updated_at,
                topics=repo_data.get("topics", []),
            )

            # Skip archived repos
            if repo_data.get("archived"):
                return None

            # Minimum stars
            if repo.stars < MIN_STARS:
                return None

            # Recent activity
            cutoff = datetime.now() - timedelta(days=MIN_RECENT_ACTIVITY_DAYS)
            if repo.updated_at < cutoff:
                return None

            # Calculate quality score
            repo.score = self._calculate_score(repo, repo_data)

            return repo

        except Exception as e:
            logger.debug("Failed to evaluate repo: %s", e)
            return None

    def _calculate_score(self, repo: DiscoveredRepo, repo_data: dict) -> float:
        """Calculate a quality score for the repository."""
        score = 0.0

        # Stars (log scale, max 40 points)
        import math

        score += min(40, math.log10(max(1, repo.stars)) * 15)

        # Recent activity (max 20 points)
        days_since_update = (datetime.now() - repo.updated_at).days
        if days_since_update < 7:
            score += 20
        elif days_since_update < 30:
            score += 15
        elif days_since_update < 90:
            score += 10
        elif days_since_update < 180:
            score += 5

        # Relevant topics (max 20 points)
        relevant_topics = {
            "claude",
            "anthropic",
            "skills",
            "agent",
            "mcp",
            "llm",
            "ai",
            "automation",
        }
        topic_matches = sum(1 for t in repo.topics if t.lower() in relevant_topics)
        score += min(20, topic_matches * 5)

        # Description quality (max 10 points)
        desc_lower = repo.description.lower()
        if any(kw in desc_lower for kw in ["skill", "agent", "claude", "anthropic"]):
            score += 10
        elif any(kw in desc_lower for kw in ["ai", "llm", "automation"]):
            score += 5

        # Forks indicate community interest (max 10 points)
        forks = repo_data.get("forks_count", 0)
        score += min(10, forks / 10)

        return round(score, 2)

    def discover(self, force_refresh: bool = False) -> list[DiscoveredRepo]:
        """Discover skill repositories from GitHub.

        Args:
            force_refresh: If True, ignore cache and search fresh.

        Returns:
            List of discovered repositories sorted by quality score.
        """
        # Add seed repos first
        for seed in SEED_REPOS:
            if seed not in self._discovered:
                owner, name = seed.split("/")
                self._discovered[seed] = DiscoveredRepo(
                    owner=owner,
                    name=name,
                    url=f"https://github.com/{seed}",
                    stars=1000,  # Seed repos get high default
                    description="Seed repository",
                    updated_at=datetime.now(),
                    topics=[],
                    score=100.0,  # High score for seeds
                )

        # Search GitHub if cache is old or force refresh
        should_search = force_refresh
        if not should_search and self.cache_path.exists():
            # Check cache age
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                    last_scan = datetime.fromisoformat(data.get("last_scan", "2000-01-01"))
                    # Refresh if older than 24 hours
                    if datetime.now() - last_scan > timedelta(hours=24):
                        should_search = True
            except Exception:
                should_search = True
        elif not self.cache_path.exists():
            should_search = True

        if should_search:
            logger.info("Searching GitHub for skill repositories...")
            for query in SEARCH_QUERIES:
                results = self._github_search(query)
                for repo_data in results:
                    repo = self._evaluate_repo(repo_data)
                    if repo and repo.full_name not in self._discovered:
                        self._discovered[repo.full_name] = repo
                        logger.debug("Discovered: %s (score: %.1f)", repo.full_name, repo.score)

            self._save_cache()

        # Return sorted by score
        return sorted(
            self._discovered.values(),
            key=lambda r: r.score,
            reverse=True,
        )

    def get_new_repos(self, registry: "RepoRegistry") -> list[DiscoveredRepo]:
        """Get repos that aren't yet tracked in the registry.

        Args:
            registry: The repo registry to check against.

        Returns:
            List of new repos not yet tracked, sorted by score.
        """
        discovered = self.discover()
        # list_all() returns list of RepoEntry objects
        tracked = {f"{r.owner}/{r.name}" for r in registry.list_all()}

        new_repos = [r for r in discovered if r.full_name not in tracked]
        logger.info(
            "Found %d new repos (out of %d discovered, %d tracked)",
            len(new_repos),
            len(discovered),
            len(tracked),
        )
        return new_repos
