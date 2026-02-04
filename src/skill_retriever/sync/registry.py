"""Registry for tracking repositories and their sync state."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RepoEntry(BaseModel):
    """A registered repository for sync tracking."""

    url: str = Field(description="Repository URL (GitHub)")
    owner: str = Field(description="Repository owner")
    name: str = Field(description="Repository name")
    last_ingested: datetime | None = Field(default=None, description="Last successful ingestion timestamp")
    last_commit_sha: str | None = Field(default=None, description="SHA of last ingested commit")
    webhook_enabled: bool = Field(default=False, description="Whether webhook is configured for this repo")
    poll_enabled: bool = Field(default=True, description="Whether to poll this repo for changes")


class RepoRegistry:
    """Registry for managing tracked repositories.

    Persists repository list and sync state to disk.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialize registry with optional custom path."""
        self._path = path or (Path.home() / ".skill-retriever" / "repo-registry.json")
        self._repos: dict[str, RepoEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if not self._path.exists():
            self._repos = {}
            return

        try:
            with self._path.open(encoding="utf-8") as f:
                data = json.load(f)

            self._repos = {}
            for key, entry_data in data.get("repos", {}).items():
                # Handle datetime serialization
                if entry_data.get("last_ingested"):
                    entry_data["last_ingested"] = datetime.fromisoformat(entry_data["last_ingested"])
                self._repos[key] = RepoEntry(**entry_data)

            logger.info("Loaded %d repos from registry", len(self._repos))
        except Exception:
            logger.exception("Failed to load registry from %s", self._path)
            self._repos = {}

    def save(self) -> None:
        """Persist registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {"repos": {}}
        for key, entry in self._repos.items():
            entry_dict = entry.model_dump()
            # Serialize datetime
            if entry_dict.get("last_ingested"):
                entry_dict["last_ingested"] = entry_dict["last_ingested"].isoformat()
            data["repos"][key] = entry_dict

        with self._path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.debug("Saved registry with %d repos", len(self._repos))

    def register(
        self,
        url: str,
        owner: str,
        name: str,
        webhook_enabled: bool = False,
        poll_enabled: bool = True,
    ) -> RepoEntry:
        """Register a repository for sync tracking."""
        key = f"{owner}/{name}"

        if key in self._repos:
            # Update existing entry
            entry = self._repos[key]
            entry.url = url
            entry.webhook_enabled = webhook_enabled
            entry.poll_enabled = poll_enabled
        else:
            entry = RepoEntry(
                url=url,
                owner=owner,
                name=name,
                webhook_enabled=webhook_enabled,
                poll_enabled=poll_enabled,
            )
            self._repos[key] = entry

        self.save()
        logger.info("Registered repo %s (webhook=%s, poll=%s)", key, webhook_enabled, poll_enabled)
        return entry

    def unregister(self, owner: str, name: str) -> bool:
        """Remove a repository from sync tracking."""
        key = f"{owner}/{name}"
        if key in self._repos:
            del self._repos[key]
            self.save()
            logger.info("Unregistered repo %s", key)
            return True
        return False

    def get(self, owner: str, name: str) -> RepoEntry | None:
        """Get a registered repository."""
        return self._repos.get(f"{owner}/{name}")

    def list_all(self) -> list[RepoEntry]:
        """List all registered repositories."""
        return list(self._repos.values())

    def list_for_polling(self) -> list[RepoEntry]:
        """List repositories that need polling (poll_enabled and not webhook_enabled)."""
        return [r for r in self._repos.values() if r.poll_enabled and not r.webhook_enabled]

    def mark_ingested(self, owner: str, name: str, commit_sha: str | None = None) -> None:
        """Update last ingestion timestamp for a repo."""
        key = f"{owner}/{name}"
        if key in self._repos:
            self._repos[key].last_ingested = datetime.now(tz=UTC)
            if commit_sha:
                self._repos[key].last_commit_sha = commit_sha
            self.save()
