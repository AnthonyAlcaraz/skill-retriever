"""External API search for skills.sh and Vercel skills directory."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SKILLSH_API_BASE = "https://skills.sh/api"
SKILLSH_TIMEOUT = 5  # seconds - don't block local results


@dataclass
class ExternalSkill:
    """A skill from an external API source."""

    owner: str
    repo: str
    skill_id: str
    name: str
    description: str
    install_count: int
    url: str


def search_skillsh(query: str, limit: int = 20) -> list[ExternalSkill]:
    """Search skills.sh API. Returns empty list on failure (non-blocking)."""
    try:
        url = f"{SKILLSH_API_BASE}/search?q={urllib.parse.quote(query)}&limit={limit}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "SkillRetriever/1.0",
        })
        with urllib.request.urlopen(req, timeout=SKILLSH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        results: list[ExternalSkill] = []
        for skill in data.get("skills", []):
            results.append(ExternalSkill(
                owner=skill.get("owner", ""),
                repo=skill.get("repo", ""),
                skill_id=skill.get("skillId", skill.get("name", "")),
                name=skill.get("name", ""),
                description=skill.get("description", ""),
                install_count=skill.get("installs", 0),
                url=f"https://skills.sh/{skill.get('owner', '')}/{skill.get('repo', '')}",
            ))
        return results
    except Exception:
        logger.debug("skills.sh API search failed for '%s'", query, exc_info=True)
        return []


def external_results_to_ranked(
    external_skills: list[ExternalSkill],
    existing_ids: set[str],
) -> tuple[list[str], dict[str, ExternalSkill]]:
    """Convert external skills to ranked ID list for RRF fusion.

    Returns:
        (ranked_ids, external_skill_map) where:
        - ranked_ids: ordered list of component IDs for RRF
        - external_skill_map: mapping of IDs to ExternalSkill for components
          NOT in local index (need on-the-fly metadata)
    """
    ranked_ids: list[str] = []
    new_skills: dict[str, ExternalSkill] = {}

    for skill in external_skills:
        # Try to match to existing local component ID
        component_id = f"{skill.owner}/{skill.repo}/skill/{skill.skill_id}"

        ranked_ids.append(component_id)
        if component_id not in existing_ids:
            # Track as external-only result
            new_skills[component_id] = skill

    return ranked_ids, new_skills
