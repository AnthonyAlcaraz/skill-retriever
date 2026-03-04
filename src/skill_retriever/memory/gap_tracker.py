"""Capability gap detection for SkillRL-inspired skill generation.

Tracks searches that return no or low-quality results, accumulating
gaps for periodic review and potential skill generation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class GapTracker:
    """Tracks capability gaps when searches return no/low results.

    Accumulates gaps for periodic review and potential skill generation.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_path = data_dir / "capability-gaps.json"
        self.gaps: list[dict[str, object]] = self._load()

    def record_gap(self, query: str, top_score: float, result_count: int) -> None:
        """Record a search that found insufficient results."""
        self.gaps.append({
            "query": query,
            "top_score": top_score,
            "result_count": result_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save()

    def get_frequent_gaps(self, min_count: int = 3) -> list[dict[str, object]]:
        """Return gap queries that appeared multiple times (candidates for new skills).

        Groups by normalized query tokens.
        """
        token_groups: dict[str, list[dict[str, object]]] = {}
        for gap in self.gaps:
            key = " ".join(sorted(str(gap["query"]).lower().split()))
            token_groups.setdefault(key, []).append(gap)

        frequent: list[dict[str, object]] = []
        for key, entries in token_groups.items():
            if len(entries) >= min_count:
                frequent.append({
                    "normalized_query": key,
                    "count": len(entries),
                    "example_queries": list(set(str(e["query"]) for e in entries))[:5],
                    "avg_top_score": sum(float(e["top_score"]) for e in entries) / len(entries),
                    "first_seen": min(str(e["timestamp"]) for e in entries),
                    "last_seen": max(str(e["timestamp"]) for e in entries),
                })

        return sorted(frequent, key=lambda x: int(x["count"]), reverse=True)

    def _load(self) -> list[dict[str, object]]:
        if self.data_path.exists():
            try:
                data = json.loads(self.data_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def save(self) -> None:
        """Persist gaps to JSON file, keeping last 1000 entries."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep last 1000 gaps to prevent unbounded growth
        if len(self.gaps) > 1000:
            self.gaps = self.gaps[-1000:]
        self.data_path.write_text(
            json.dumps(self.gaps, indent=2),
            encoding="utf-8",
        )
