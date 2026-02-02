"""Git signal extraction for component freshness and activity metrics."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003
from typing import Any

from git import InvalidGitRepositoryError, Repo

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "last_updated": None,
    "commit_count": 0,
    "commit_frequency_30d": 0.0,
}


def extract_git_signals(repo_path: Path, file_relative_path: str) -> dict[str, Any]:
    """Extract git activity metrics for a file within a repository.

    Returns a dict with keys: last_updated, commit_count, commit_frequency_30d.
    Falls back to defaults when the path is not a git repository.
    """
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        logger.debug("Not a git repository: %s", repo_path)
        return dict(_DEFAULTS)

    commits = list(repo.iter_commits(paths=file_relative_path, max_count=500))

    if not commits:
        return dict(_DEFAULTS)

    last_updated = commits[0].committed_datetime
    commit_count = len(commits)

    # Count commits in the last 30 days
    now = datetime.now(tz=UTC)
    thirty_days_ago = now.timestamp() - (30 * 24 * 60 * 60)
    recent_commits = sum(
        1 for c in commits if c.committed_date >= thirty_days_ago
    )
    commit_frequency_30d = round(recent_commits / 30.0, 2)

    return {
        "last_updated": last_updated,
        "commit_count": commit_count,
        "commit_frequency_30d": commit_frequency_30d,
    }
