"""Tests for FAISSVectorStore."""

from __future__ import annotations

import numpy as np
import pytest

from skill_retriever.memory.vector_store import FAISSVectorStore

DIM = 384


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _random_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal((n, DIM)).astype(np.float32)


# ------------------------------------------------------------------
# 1. test_add_and_count
# ------------------------------------------------------------------


def test_add_and_count() -> None:
    rng = _rng()
    store = FAISSVectorStore(dimensions=DIM)
    vecs = _random_vectors(rng, 3)
    for i in range(3):
        store.add(f"comp-{i}", vecs[i])
    assert store.count == 3


# ------------------------------------------------------------------
# 2. test_add_and_search
# ------------------------------------------------------------------


def test_add_and_search() -> None:
    rng = _rng()
    store = FAISSVectorStore(dimensions=DIM)
    vecs = _random_vectors(rng, 5)
    for i in range(5):
        store.add(f"comp-{i}", vecs[i])

    # Query with the exact same vector as comp-2
    results = store.search(vecs[2], top_k=3)
    assert len(results) >= 1
    top_id, top_sim = results[0]
    assert top_id == "comp-2"
    assert top_sim > 0.99


# ------------------------------------------------------------------
# 3. test_search_returns_cosine_similarity
# ------------------------------------------------------------------


def test_search_returns_cosine_similarity() -> None:
    store = FAISSVectorStore(dimensions=DIM)
    # Create two orthogonal vectors (zero cosine similarity)
    a = np.zeros(DIM, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(DIM, dtype=np.float32)
    b[1] = 1.0

    store.add("parallel", a)
    store.add("orthogonal", b)

    results = store.search(a, top_k=2)
    # First result should be the parallel vector with sim ~1.0
    assert results[0][0] == "parallel"
    assert results[0][1] > 0.99
    # Second result should be orthogonal with sim ~0.0
    assert results[1][0] == "orthogonal"
    assert abs(results[1][1]) < 0.01


# ------------------------------------------------------------------
# 4. test_search_filters_negative_one
# ------------------------------------------------------------------


def test_search_filters_negative_one() -> None:
    rng = _rng()
    store = FAISSVectorStore(dimensions=DIM)
    vecs = _random_vectors(rng, 2)
    store.add("a", vecs[0])
    store.add("b", vecs[1])

    # Ask for more results than vectors exist
    results = store.search(vecs[0], top_k=5)
    assert len(results) == 2


# ------------------------------------------------------------------
# 5. test_add_batch
# ------------------------------------------------------------------


def test_add_batch() -> None:
    rng = _rng()
    store = FAISSVectorStore(dimensions=DIM)
    ids = [f"batch-{i}" for i in range(10)]
    vecs = _random_vectors(rng, 10)

    store.add_batch(ids, vecs)
    assert store.count == 10

    results = store.search(vecs[3], top_k=1)
    assert results[0][0] == "batch-3"


# ------------------------------------------------------------------
# 6. test_remove
# ------------------------------------------------------------------


def test_remove() -> None:
    rng = _rng()
    store = FAISSVectorStore(dimensions=DIM)
    vecs = _random_vectors(rng, 3)
    for i in range(3):
        store.add(f"comp-{i}", vecs[i])

    store.remove("comp-1")
    assert store.count == 2
    assert not store.contains("comp-1")
    assert store.contains("comp-0")
    assert store.contains("comp-2")


# ------------------------------------------------------------------
# 7. test_remove_nonexistent_raises
# ------------------------------------------------------------------


def test_remove_nonexistent_raises() -> None:
    store = FAISSVectorStore(dimensions=DIM)
    with pytest.raises(KeyError):
        store.remove("does-not-exist")


# ------------------------------------------------------------------
# 8. test_save_and_load
# ------------------------------------------------------------------


def test_save_and_load(tmp_path: str) -> None:
    rng = _rng()
    store = FAISSVectorStore(dimensions=DIM)
    vecs = _random_vectors(rng, 5)
    for i in range(5):
        store.add(f"comp-{i}", vecs[i])

    store.save(str(tmp_path))

    loaded = FAISSVectorStore(dimensions=DIM)
    loaded.load(str(tmp_path))

    assert loaded.count == 5
    results = loaded.search(vecs[2], top_k=1)
    assert results[0][0] == "comp-2"
    assert results[0][1] > 0.99


# ------------------------------------------------------------------
# 9. test_contains
# ------------------------------------------------------------------


def test_contains() -> None:
    rng = _rng()
    store = FAISSVectorStore(dimensions=DIM)
    store.add("present", _random_vectors(rng, 1)[0])

    assert store.contains("present")
    assert not store.contains("absent")
