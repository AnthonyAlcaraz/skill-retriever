"""Auto-heal system for failed ingestions.

Monitors ingestion failures, diagnoses issues, and attempts automatic recovery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skill_retriever.sync.registry import RepoRegistry

logger = logging.getLogger(__name__)


class FailureType(StrEnum):
    """Types of ingestion failures."""

    CLONE_FAILED = "clone_failed"
    NO_COMPONENTS = "no_components"
    PARSE_ERROR = "parse_error"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    STRATEGY_MISMATCH = "strategy_mismatch"
    UNKNOWN = "unknown"


@dataclass
class IngestionFailure:
    """Record of a failed ingestion attempt."""

    repo_key: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    retry_count: int = 0
    resolved: bool = False
    resolution: str = ""


@dataclass
class HealingState:
    """State of the auto-healing system."""

    failures: dict[str, IngestionFailure] = field(default_factory=dict)
    last_heal_attempt: datetime | None = None
    total_healed: int = 0
    total_failed: int = 0


class AutoHealer:
    """Monitors and heals ingestion failures."""

    # Max retry attempts before giving up
    MAX_RETRIES = 3

    # Repos that should be skipped (known incompatible)
    SKIP_REPOS: set[str] = set()

    def __init__(
        self,
        state_path: Path | None = None,
    ) -> None:
        """Initialize the auto-healer.

        Args:
            state_path: Path to persist healing state.
        """
        self.state_path = state_path or (
            Path.home() / ".skill-retriever" / "heal-state.json"
        )
        self.state = HealingState()
        self._load_state()

    def _load_state(self) -> None:
        """Load healing state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    data = json.load(f)

                self.state.total_healed = data.get("total_healed", 0)
                self.state.total_failed = data.get("total_failed", 0)

                if data.get("last_heal_attempt"):
                    self.state.last_heal_attempt = datetime.fromisoformat(
                        data["last_heal_attempt"]
                    )

                for failure_data in data.get("failures", []):
                    failure = IngestionFailure(
                        repo_key=failure_data["repo_key"],
                        failure_type=FailureType(failure_data["failure_type"]),
                        error_message=failure_data["error_message"],
                        timestamp=datetime.fromisoformat(failure_data["timestamp"]),
                        retry_count=failure_data.get("retry_count", 0),
                        resolved=failure_data.get("resolved", False),
                        resolution=failure_data.get("resolution", ""),
                    )
                    self.state.failures[failure.repo_key] = failure

                logger.info(
                    "Loaded healing state: %d failures, %d healed",
                    len(self.state.failures),
                    self.state.total_healed,
                )
            except Exception as e:
                logger.warning("Failed to load healing state: %s", e)

    def _save_state(self) -> None:
        """Save healing state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_healed": self.state.total_healed,
            "total_failed": self.state.total_failed,
            "last_heal_attempt": (
                self.state.last_heal_attempt.isoformat()
                if self.state.last_heal_attempt
                else None
            ),
            "failures": [
                {
                    "repo_key": f.repo_key,
                    "failure_type": f.failure_type.value,
                    "error_message": f.error_message,
                    "timestamp": f.timestamp.isoformat(),
                    "retry_count": f.retry_count,
                    "resolved": f.resolved,
                    "resolution": f.resolution,
                }
                for f in self.state.failures.values()
            ],
        }

        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_failure(
        self,
        repo_key: str,
        error_message: str,
        failure_type: FailureType | None = None,
    ) -> None:
        """Record a new ingestion failure.

        Args:
            repo_key: The repository that failed (owner/name).
            error_message: The error message from the failure.
            failure_type: Optional explicit failure type (auto-detected if not provided).
        """
        # Auto-detect failure type from error message
        if failure_type is None:
            failure_type = self._diagnose_failure(error_message)

        # Check if this is a retry
        existing = self.state.failures.get(repo_key)
        retry_count = (existing.retry_count + 1) if existing else 0

        failure = IngestionFailure(
            repo_key=repo_key,
            failure_type=failure_type,
            error_message=error_message[:500],  # Truncate long messages
            timestamp=datetime.now(),
            retry_count=retry_count,
        )

        self.state.failures[repo_key] = failure
        self.state.total_failed += 1
        self._save_state()

        logger.warning(
            "Recorded failure for %s: %s (retry %d)",
            repo_key,
            failure_type.value,
            retry_count,
        )

    def _diagnose_failure(self, error_message: str) -> FailureType:
        """Diagnose failure type from error message."""
        error_lower = error_message.lower()

        if any(kw in error_lower for kw in ["clone", "git", "repository not found"]):
            return FailureType.CLONE_FAILED

        if any(kw in error_lower for kw in ["no components", "0 components"]):
            return FailureType.NO_COMPONENTS

        if any(kw in error_lower for kw in ["parse", "yaml", "frontmatter", "json"]):
            return FailureType.PARSE_ERROR

        if any(kw in error_lower for kw in ["network", "connection", "timeout"]):
            return FailureType.NETWORK_ERROR

        if any(kw in error_lower for kw in ["rate limit", "403", "too many requests"]):
            return FailureType.RATE_LIMITED

        if any(kw in error_lower for kw in ["strategy", "no strategy"]):
            return FailureType.STRATEGY_MISMATCH

        return FailureType.UNKNOWN

    def get_healable_failures(self) -> list[IngestionFailure]:
        """Get failures that can potentially be healed.

        Returns:
            List of failures that haven't exceeded retry limit and aren't resolved.
        """
        healable = []
        for failure in self.state.failures.values():
            if failure.resolved:
                continue
            if failure.retry_count >= self.MAX_RETRIES:
                continue
            if failure.repo_key in self.SKIP_REPOS:
                continue
            healable.append(failure)

        return sorted(healable, key=lambda f: f.timestamp)

    def mark_healed(self, repo_key: str, resolution: str) -> None:
        """Mark a failure as healed.

        Args:
            repo_key: The repository that was healed.
            resolution: Description of how it was resolved.
        """
        if repo_key in self.state.failures:
            failure = self.state.failures[repo_key]
            failure.resolved = True
            failure.resolution = resolution
            self.state.total_healed += 1
            self._save_state()

            logger.info("Marked %s as healed: %s", repo_key, resolution)

    def get_healing_suggestions(self, failure: IngestionFailure) -> list[str]:
        """Get suggestions for healing a failure.

        Args:
            failure: The failure to get suggestions for.

        Returns:
            List of suggested actions to heal the failure.
        """
        suggestions = []

        if failure.failure_type == FailureType.CLONE_FAILED:
            suggestions.extend([
                "Check if repository still exists on GitHub",
                "Verify repository is public or token has access",
                "Check for network connectivity issues",
            ])

        elif failure.failure_type == FailureType.NO_COMPONENTS:
            suggestions.extend([
                "Repository may use a non-standard structure",
                "Check if AwesomeListStrategy should apply (README-based)",
                "Check if PluginMarketplaceStrategy should apply (plugins/ dir)",
                "Repository may be a meta-list without actual skill files",
            ])

        elif failure.failure_type == FailureType.PARSE_ERROR:
            suggestions.extend([
                "Check for malformed YAML frontmatter in files",
                "Verify file encoding is UTF-8",
                "Check for syntax errors in SKILL.md files",
            ])

        elif failure.failure_type == FailureType.RATE_LIMITED:
            suggestions.extend([
                "Wait and retry later",
                "Add GitHub API token for higher limits",
                "Reduce polling frequency",
            ])

        elif failure.failure_type == FailureType.STRATEGY_MISMATCH:
            suggestions.extend([
                "Repository structure not recognized",
                "May need to add a new extraction strategy",
                "Check if repo follows Agent Skills specification",
            ])

        else:
            suggestions.append("Check logs for detailed error information")

        return suggestions

    def get_status(self) -> dict:
        """Get current healing status.

        Returns:
            Dictionary with healing statistics.
        """
        failures = self.state.failures
        unresolved = [f for f in failures.values() if not f.resolved]
        resolved = [f for f in failures.values() if f.resolved]

        by_type = {}
        for f in unresolved:
            by_type[f.failure_type.value] = by_type.get(f.failure_type.value, 0) + 1

        return {
            "total_failures": len(failures),
            "unresolved": len(unresolved),
            "resolved": len(resolved),
            "total_healed": self.state.total_healed,
            "healable": len(self.get_healable_failures()),
            "by_type": by_type,
            "last_heal_attempt": (
                self.state.last_heal_attempt.isoformat()
                if self.state.last_heal_attempt
                else None
            ),
        }
