"""Shared test fixtures for skill-retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def davila7_repo() -> Path:
    """Path to the davila7-style test fixture repository."""
    return FIXTURES_DIR / "davila7_sample"


@pytest.fixture
def flat_repo() -> Path:
    """Path to the flat .claude/ test fixture repository."""
    return FIXTURES_DIR / "flat_sample"
