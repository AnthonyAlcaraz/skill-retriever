"""Ingestion cache for incremental updates (SYNC-03)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class IngestionCache:
    """Tracks file content hashes to enable incremental ingestion.

    For each repo, stores a mapping of component_id -> content_hash.
    On re-ingestion, components with unchanged hashes are skipped.
    """

    def __init__(self, cache_path: Path) -> None:
        """Initialize the ingestion cache.

        Args:
            cache_path: Path to JSON file for persistence.
        """
        self.cache_path = cache_path
        self._hashes: dict[str, dict[str, str]] = {}  # repo_key -> {component_id -> hash}
        self._load()

    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if self.cache_path.exists():
            try:
                self._hashes = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._hashes = {}

    def save(self) -> None:
        """Persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._hashes, indent=2), encoding="utf-8"
        )

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def get_repo_hashes(self, repo_key: str) -> dict[str, str]:
        """Get all hashes for a repository."""
        return self._hashes.get(repo_key, {})

    def is_unchanged(self, repo_key: str, component_id: str, content: str) -> bool:
        """Check if component content matches cached hash."""
        repo_hashes = self._hashes.get(repo_key, {})
        cached_hash = repo_hashes.get(component_id)
        if cached_hash is None:
            return False
        return cached_hash == self.compute_hash(content)

    def update_hash(self, repo_key: str, component_id: str, content: str) -> None:
        """Update the hash for a component."""
        if repo_key not in self._hashes:
            self._hashes[repo_key] = {}
        self._hashes[repo_key][component_id] = self.compute_hash(content)

    def clear_repo(self, repo_key: str) -> None:
        """Clear all hashes for a repository (for full rebuild)."""
        self._hashes.pop(repo_key, None)
