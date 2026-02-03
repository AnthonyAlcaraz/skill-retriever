"""FAISS-backed vector store with cosine similarity via IndexFlatIP."""

from __future__ import annotations

import json
from pathlib import Path

import faiss  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from skill_retriever.config import EMBEDDING_CONFIG

_INDEX_FILENAME = "faiss_index.bin"
_MAPPING_FILENAME = "id_mapping.json"


class FAISSVectorStore:
    """Vector store using FAISS IndexFlatIP with string-to-int ID mapping.

    Vectors are L2-normalized before insertion so that inner-product
    scores equal cosine similarity.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        dim = dimensions if dimensions is not None else EMBEDDING_CONFIG.dimensions
        base = faiss.IndexFlatIP(dim)
        self._index: faiss.IndexIDMap = faiss.IndexIDMap(base)
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_id: int = 0
        self._dimensions: int = dim

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(self, component_id: str, embedding: np.ndarray) -> None:
        """Add a single vector, assigning the next integer ID."""
        int_id = self._next_id
        self._next_id += 1
        self._id_to_int[component_id] = int_id
        self._int_to_id[int_id] = component_id

        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)  # pyright: ignore[reportUnknownMemberType]
        self._index.add_with_ids(vec, np.array([int_id], dtype=np.int64))  # pyright: ignore[reportUnknownMemberType, reportCallIssue]

    def add_batch(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Bulk-add vectors in a single FAISS call."""
        vecs = np.asarray(embeddings, dtype=np.float32).reshape(len(ids), -1)
        faiss.normalize_L2(vecs)  # pyright: ignore[reportUnknownMemberType]

        int_ids = np.empty(len(ids), dtype=np.int64)
        for i, cid in enumerate(ids):
            int_id = self._next_id
            self._next_id += 1
            self._id_to_int[cid] = int_id
            self._int_to_id[int_id] = cid
            int_ids[i] = int_id

        self._index.add_with_ids(vecs, int_ids)  # pyright: ignore[reportUnknownMemberType, reportCallIssue]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Return up to *top_k* ``(component_id, similarity)`` pairs."""
        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)  # pyright: ignore[reportUnknownMemberType]

        distances, indices = self._index.search(vec, top_k)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportCallIssue]

        results: list[tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0], strict=True):  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            if idx == -1:
                continue
            results.append((self._int_to_id[int(idx)], float(dist)))  # pyright: ignore[reportUnknownArgumentType]
        return results

    # ------------------------------------------------------------------
    # Remove / query
    # ------------------------------------------------------------------

    def remove(self, component_id: str) -> None:
        """Remove a vector by its string ID. Raises ``KeyError`` if absent."""
        if component_id not in self._id_to_int:
            msg = f"Component ID not found: {component_id}"
            raise KeyError(msg)
        int_id = self._id_to_int.pop(component_id)
        del self._int_to_id[int_id]
        self._index.remove_ids(np.array([int_id], dtype=np.int64))  # pyright: ignore[reportUnknownMemberType]

    @property
    def count(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    def contains(self, component_id: str) -> bool:
        """Check whether *component_id* has been added."""
        return component_id in self._id_to_int

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Write FAISS index and ID mapping to *directory*."""
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(dirpath / _INDEX_FILENAME))  # pyright: ignore[reportUnknownMemberType]

        mapping = {
            "id_to_int": self._id_to_int,
            "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
            "next_id": self._next_id,
        }
        (dirpath / _MAPPING_FILENAME).write_text(json.dumps(mapping), encoding="utf-8")

    def load(self, directory: str) -> None:
        """Restore FAISS index and ID mapping from *directory*."""
        dirpath = Path(directory)

        self._index = faiss.read_index(str(dirpath / _INDEX_FILENAME))  # pyright: ignore[reportUnknownMemberType]

        raw = json.loads((dirpath / _MAPPING_FILENAME).read_text(encoding="utf-8"))
        self._id_to_int = raw["id_to_int"]
        self._int_to_id = {int(k): v for k, v in raw["int_to_id"].items()}
        self._next_id = raw["next_id"]
