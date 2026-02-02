"""Smoke tests for Phase 01 success criteria."""

import importlib


def test_version_importable() -> None:
    """SC-1: from skill_retriever import __version__ works."""
    from skill_retriever import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_embedding_config_loadable() -> None:
    """SC-4: Embedding model version is pinned and loadable."""
    from skill_retriever.config import EMBEDDING_CONFIG

    assert EMBEDDING_CONFIG.model_name == "BAAI/bge-small-en-v1.5"
    assert EMBEDDING_CONFIG.dimensions == 384
    assert EMBEDDING_CONFIG.max_length == 512


def test_subpackages_importable() -> None:
    """All Iusztin virtual layer subpackages are importable."""
    subpackages = [
        "skill_retriever.entities",
        "skill_retriever.memory",
        "skill_retriever.mcp",
        "skill_retriever.models",
        "skill_retriever.nodes",
        "skill_retriever.utils",
        "skill_retriever.workflows",
    ]
    for pkg in subpackages:
        mod = importlib.import_module(pkg)
        assert mod is not None
