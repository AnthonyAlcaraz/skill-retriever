"""Two-phase entity resolution pipeline for deduplicating components."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from rapidfuzz import fuzz

if TYPE_CHECKING:
    from fastembed import TextEmbedding

    from skill_retriever.entities import ComponentMetadata


class EntityResolver:
    """Deduplicates ComponentMetadata entities using fuzzy matching and embeddings.

    Phase 1 uses fuzzy string matching (token_sort_ratio) to find candidate
    duplicate pairs within the same component_type.

    Phase 2 (optional) confirms candidates via embedding cosine similarity.
    When no embedding_model is provided, all Phase 1 candidates are treated
    as confirmed duplicates.
    """

    def __init__(
        self,
        embedding_model: TextEmbedding | None = None,
        fuzzy_threshold: float = 80.0,
        embedding_threshold: float = 0.85,
    ) -> None:
        self._embedding_model = embedding_model
        self._fuzzy_threshold = fuzzy_threshold
        self._embedding_threshold = embedding_threshold

    def resolve(self, entities: list[ComponentMetadata]) -> list[ComponentMetadata]:
        """Resolve duplicates across a list of entities.

        Groups entities by component_type, finds duplicates within each group,
        merges duplicate groups, and returns the deduplicated list.
        """
        if not entities:
            return []

        # Group by component_type (blocking strategy)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, entity in enumerate(entities):
            groups[entity.component_type.value].append(idx)

        # Collect all confirmed duplicate pairs across all type groups
        all_confirmed_pairs: list[tuple[int, int]] = []

        for _ctype, indices in groups.items():
            if len(indices) < 2:
                continue

            group_entities = [entities[i] for i in indices]

            # Phase 1: fuzzy candidates
            candidates = self._find_fuzzy_candidates(group_entities)
            if not candidates:
                continue

            # Phase 2: embedding confirmation (or skip)
            if self._embedding_model is not None:
                confirmed_local = self._confirm_with_embeddings(
                    group_entities, candidates
                )
            else:
                # Fuzzy-only mode: all candidates are confirmed
                confirmed_local = [(i, j) for i, j, _score in candidates]

            # Map local indices back to global indices
            for li, lj in confirmed_local:
                all_confirmed_pairs.append((indices[li], indices[lj]))

        if not all_confirmed_pairs:
            return list(entities)

        # Union-Find to group transitive duplicates
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, j in all_confirmed_pairs:
            union(i, j)

        # Build groups from Union-Find
        merge_groups: dict[int, list[int]] = defaultdict(list)
        all_involved: set[int] = set()
        for i, j in all_confirmed_pairs:
            all_involved.add(i)
            all_involved.add(j)

        for idx in all_involved:
            root = find(idx)
            merge_groups[root].append(idx)

        # Deduplicate: merge groups, keep non-duplicates as-is
        merged_indices: set[int] = set()
        result: list[ComponentMetadata] = []

        for _root, member_indices in merge_groups.items():
            group_members = [entities[i] for i in member_indices]
            result.append(self._merge_group(group_members))
            merged_indices.update(member_indices)

        # Add entities that were not part of any duplicate group
        for idx, entity in enumerate(entities):
            if idx not in merged_indices:
                result.append(entity)

        return result

    def _find_fuzzy_candidates(
        self, entities: list[ComponentMetadata]
    ) -> list[tuple[int, int, float]]:
        """Find candidate duplicate pairs using fuzzy string matching.

        Returns (idx_i, idx_j, score) triples for pairs scoring above
        the fuzzy threshold.
        """
        candidates: list[tuple[int, int, float]] = []
        n = len(entities)
        for i in range(n):
            for j in range(i + 1, n):
                score = fuzz.token_sort_ratio(entities[i].name, entities[j].name)
                if score >= self._fuzzy_threshold:
                    candidates.append((i, j, score))
        return candidates

    def _confirm_with_embeddings(
        self,
        entities: list[ComponentMetadata],
        candidates: list[tuple[int, int, float]],
    ) -> list[tuple[int, int]]:
        """Confirm candidate pairs using embedding cosine similarity.

        For each candidate pair, computes cosine similarity between
        "{name} {description}" embeddings. Returns pairs above the
        embedding threshold.
        """
        if self._embedding_model is None:
            return [(i, j) for i, j, _s in candidates]

        # Collect unique indices that need embeddings
        needed_indices: set[int] = set()
        for i, j, _s in candidates:
            needed_indices.add(i)
            needed_indices.add(j)

        # Build texts and compute embeddings in batch
        idx_list = sorted(needed_indices)
        texts = [
            f"{entities[i].name} {entities[i].description}" for i in idx_list
        ]
        embeddings_iter = self._embedding_model.embed(texts)
        embeddings_list = list(embeddings_iter)

        # Map index -> embedding
        idx_to_emb: dict[int, np.ndarray] = {}
        for pos, idx in enumerate(idx_list):
            idx_to_emb[idx] = np.array(embeddings_list[pos], dtype=np.float64)

        # Confirm pairs
        confirmed: list[tuple[int, int]] = []
        for i, j, _score in candidates:
            emb_i = idx_to_emb[i]
            emb_j = idx_to_emb[j]
            norm_i = np.linalg.norm(emb_i)
            norm_j = np.linalg.norm(emb_j)
            if norm_i == 0.0 or norm_j == 0.0:
                continue
            cosine_sim = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))
            if cosine_sim >= self._embedding_threshold:
                confirmed.append((i, j))

        return confirmed

    @staticmethod
    def _merge_group(entities: list[ComponentMetadata]) -> ComponentMetadata:
        """Merge a group of duplicate entities, keeping the richest metadata.

        Richness is determined by: longest description, most tags, most tools.
        Tags and tools are unioned across all entities. The most recent
        last_updated timestamp is kept.
        """
        if len(entities) == 1:
            return entities[0]

        # Score each entity for richness
        def richness(e: ComponentMetadata) -> tuple[int, int, int]:
            return (len(e.description), len(e.tags), len(e.tools))

        # Sort by richness descending and pick the best as base
        ranked = sorted(entities, key=richness, reverse=True)
        base = ranked[0]

        # Union tags and tools from all entities
        all_tags: set[str] = set()
        all_tools: set[str] = set()
        latest_updated = base.last_updated

        for entity in entities:
            all_tags.update(entity.tags)
            all_tools.update(entity.tools)
            if (
                entity.last_updated is not None
                and (latest_updated is None or entity.last_updated > latest_updated)
            ):
                latest_updated = entity.last_updated

        return base.model_copy(
            update={
                "tags": sorted(all_tags),
                "tools": sorted(all_tools),
                "last_updated": latest_updated,
            }
        )
