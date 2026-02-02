# Phase 01: Foundation - Research

**Researched:** 2026-02-02
**Domain:** Python project scaffolding, packaging, dev tooling, embedding model configuration
**Confidence:** HIGH

## Summary

Phase 01 establishes the project skeleton: a `uv`-managed Python package using `src` layout, with all dependencies resolved, dev tools (Ruff, pyright, pytest) configured, and the FastEmbed embedding model pinned. The project must be importable (`from skill_retriever import __version__`) and pass all linting/type-checking with zero warnings from day one.

The standard approach is `uv init --lib` which generates the correct `src/skill_retriever/` layout with `uv_build` as the build backend. This is a single-plan phase: one pyproject.toml, one directory tree, one config file for the embedding model, one empty test suite.

**Primary recommendation:** Use `uv init --lib` to scaffold, then manually expand the `src/skill_retriever/` directory with Iusztin virtual layer subdirectories (empty `__init__.py` files) and a `config.py` for embedding model pinning.

## Standard Stack

### Core (Phase 01 scope)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| uv | >=0.9.28 | Package/project manager | Rust-powered, native pyproject.toml, lockfile. Default for new Python projects in 2026. |
| uv_build | >=0.9.28,<0.10.0 | Build backend | Default backend for `uv init --lib`. Tight integration with uv, fastest wheel builds. |
| fastembed | >=0.7.4 | Embedding model | ONNX Runtime, no PyTorch. 384-dim bge-small-en-v1.5. ~50MB install. |
| pydantic | >=2.12.5 | Config validation | Used for config model that holds embedding model name + dimensions. |

### Dev Tools (Phase 01 scope)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ruff | >=0.14.14 | Linter + formatter | Every commit. Replaces Flake8 + Black + isort. |
| pyright | >=1.1.408 | Type checker | Every commit. Strict mode from day one. |
| pytest | >=9.0.2 | Test framework | `uv run pytest` must pass with empty suite. |
| pytest-asyncio | >=0.24 | Async test support | MCP tools are async. Auto mode configured. |
| pytest-cov | >=6.0 | Coverage | Track from Phase 01 onward. |

### Deferred (NOT Phase 01)

| Library | Version | Purpose | Phase |
|---------|---------|---------|-------|
| kuzu | ==0.11.3 | Graph database | Phase 03 |
| fastmcp | >=2.0,<3.0 | MCP server | Phase 06 |
| scikit-network | >=0.33.0 | PPR computation | Phase 04 |

**Installation (Phase 01 only):**
```bash
uv init --lib skill-retriever
cd skill-retriever

# Core (needed at import time for config)
uv add "fastembed>=0.7.4"
uv add "pydantic>=2.12.5"

# Dev tools
uv add --dev "pytest>=9.0.2"
uv add --dev "pytest-asyncio>=0.24"
uv add --dev "pytest-cov>=6.0"
uv add --dev "ruff>=0.14.14"
uv add --dev "pyright>=1.1.408"
```

**Note on deferred dependencies:** Do NOT add kuzu, fastmcp, or scikit-network in Phase 01. They add install weight and are unused until later phases. Add them when their phase starts.

## Architecture Patterns

### Recommended Project Structure

```
skill-retriever/
├── .python-version          # "3.13" — created by uv init
├── pyproject.toml           # Single source of truth for project config
├── README.md
├── src/
│   └── skill_retriever/
│       ├── __init__.py      # __version__, lazy embedding config
│       ├── py.typed          # PEP 561 marker (created by uv init --lib)
│       ├── config.py         # EmbeddingConfig Pydantic model + defaults
│       ├── entities/         # Domain layer: Pydantic models (Phase 02)
│       │   └── __init__.py
│       ├── nodes/            # Domain layer: AI logic units (Phase 03-04)
│       │   └── __init__.py
│       ├── workflows/        # Application layer: orchestration (Phase 05)
│       │   └── __init__.py
│       ├── models/           # Infrastructure: LLM/embedding providers (Phase 03)
│       │   └── __init__.py
│       ├── memory/           # Infrastructure: graph/vector stores (Phase 03)
│       │   └── __init__.py
│       ├── mcp/              # Serving layer: FastMCP server (Phase 06)
│       │   └── __init__.py
│       └── utils/            # Shared utilities
│           └── __init__.py
└── tests/
    ├── __init__.py
    └── test_init.py          # Smoke test: version import, config load
```

### Pattern 1: uv init --lib for MCP Server Package

**What:** Use `uv init --lib` to create a library-style package that will also serve as an MCP server. The MCP server entry point lives under `src/skill_retriever/mcp/` but the package itself is a standard installable library.

**When to use:** Always for Python MCP servers. The `src` layout ensures the installed package is tested as an installed package (not source tree), and FastMCP expects an importable module.

**How it works:**
```bash
uv init --lib skill-retriever
# Creates:
# skill-retriever/
# ├── pyproject.toml        (with [build-system] using uv_build)
# ├── src/skill_retriever/
# │   ├── __init__.py       (def hello() -> str: ...)
# │   └── py.typed
# └── .python-version       ("3.13")
```

uv_build auto-discovers `src/skill_retriever/` as the package module. No explicit `[tool.uv.build-backend]` config is needed if the package name matches the directory name (with dashes converted to underscores).

**Source:** [uv docs — Creating projects](https://docs.astral.sh/uv/concepts/projects/init/), [uv docs — Build backend](https://docs.astral.sh/uv/concepts/build-backend/)

### Pattern 2: Iusztin Virtual Layers as Flat Subdirectories

**What:** The four conceptual layers (Domain, Application, Infrastructure, Serving) map to flat subdirectories under `src/skill_retriever/`. No physical `domain/`, `application/`, `infrastructure/` directories.

**When to use:** All Python AI projects per CLAUDE.md rules.

**Mapping:**
| Virtual Layer | Directories | Phase Created |
|---------------|-------------|---------------|
| Domain | `entities/`, `nodes/` | 01 (empty), 02 (populated) |
| Application | `workflows/` | 01 (empty), 05 (populated) |
| Infrastructure | `models/`, `memory/` | 01 (empty), 03 (populated) |
| Serving | `mcp/` | 01 (empty), 06 (populated) |
| Shared | `utils/` | 01 (empty), as needed |

**Rule:** Create all directories in Phase 01 with empty `__init__.py` files. This ensures imports work from day one and pyright can validate the full package structure.

### Pattern 3: Pydantic Config for Embedding Model Pinning

**What:** A Pydantic `BaseModel` in `config.py` that holds the embedding model name, dimensions, and cache directory. Loaded once at import time, overridable via environment variables.

**Why Pydantic for config:** Validates types at load time, serializes to/from JSON, supports env var overrides via `model_config`. Already a dependency.

**Example:**
```python
# src/skill_retriever/config.py
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """Pinned embedding model configuration."""

    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model ID for FastEmbed",
    )
    dimensions: int = Field(
        default=384,
        description="Embedding vector dimensions (must match model)",
    )
    max_length: int = Field(
        default=512,
        description="Maximum token length for embedding input",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Model cache directory (None = FastEmbed default)",
    )

    model_config = {"env_prefix": "SKILL_RETRIEVER_"}
```

**Note:** This uses `pydantic-settings` for env var support. If we want to avoid the extra dependency, use plain `BaseModel` and load from a JSON/TOML file instead. Decision for planner: plain `BaseModel` with a TOML/JSON config file is simpler and avoids the dependency. Recommendation: use plain `BaseModel` + a `config.toml` or constants in `config.py`.

**Simpler alternative (no extra dependency):**
```python
# src/skill_retriever/config.py
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Pinned embedding model configuration."""

    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    dimensions: int = Field(default=384)
    max_length: int = Field(default=512)
    cache_dir: str | None = Field(default=None)


# Singleton — the pinned config
EMBEDDING_CONFIG = EmbeddingConfig()
```

**Recommendation:** Use the simpler alternative. No pydantic-settings dependency. Env var overrides can be added in Phase 06 if needed.

### Anti-Patterns to Avoid

- **Physical layer directories:** Do NOT create `domain/entities/` or `infrastructure/memory/`. Flat structure under `src/skill_retriever/`.
- **Deferred directory creation:** Create ALL subdirectories in Phase 01. If they exist from day one, pyright validates imports across the full tree.
- **Version in pyproject.toml only:** The `__version__` must also be importable from Python. Use `importlib.metadata` to read it from installed package metadata (the canonical approach since PEP 517).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Package version access | Hard-coded string in `__init__.py` | `importlib.metadata.version("skill-retriever")` | Stays in sync with pyproject.toml automatically. Single source of truth. |
| Build system | setup.py / setuptools | uv_build | Default for uv projects. Faster. Less config. |
| Import sorting | Manual / isort standalone | Ruff `select = ["I"]` | Ruff replaces isort. One tool. |
| Code formatting | Black standalone | `ruff format` | Ruff replaces Black. One tool. |
| Config validation | Dict/dataclass with manual checks | Pydantic BaseModel | Type validation, serialization, defaults. Already a dependency. |

**Key insight:** Phase 01 is pure infrastructure. Every decision here either saves or costs time in all 6 subsequent phases. Get the toolchain right now.

## Common Pitfalls

### Pitfall 1: Forgetting the build system for src layout

**What goes wrong:** Without `[build-system]` in pyproject.toml, `uv run` cannot install the package into the environment. `from skill_retriever import __version__` fails with ModuleNotFoundError.
**Why it happens:** `uv init` (without `--lib`) creates an app without build system. The `src/` layout requires installation to be importable.
**How to avoid:** Always use `uv init --lib`. Verify `[build-system]` section exists in pyproject.toml.
**Warning signs:** `ModuleNotFoundError: No module named 'skill_retriever'` when running tests.

### Pitfall 2: pyright strict mode on third-party libraries without stubs

**What goes wrong:** pyright strict reports `reportMissingTypeStubs` for libraries like fastembed, kuzu that ship without `py.typed` or full stubs.
**Why it happens:** Strict mode requires type information for all imports.
**How to avoid:** Set `reportMissingTypeStubs = false` in pyright config. These libraries have partial or no stubs. Alternatively use `reportUnknownMemberType = false` selectively.
**Warning signs:** Hundreds of pyright errors on first run, all from third-party imports.

**Recommended pyright overrides for this project:**
```toml
[tool.pyright]
pythonVersion = "3.13"
typeCheckingMode = "strict"
reportMissingTypeStubs = false
```

### Pitfall 3: pytest-asyncio mode not configured

**What goes wrong:** Async tests silently don't run, or require manual `@pytest.mark.asyncio` on every test.
**Why it happens:** Default mode is "strict" — async tests must be explicitly marked.
**How to avoid:** Set `asyncio_mode = "auto"` in `[tool.pytest.ini_options]`.
**Warning signs:** Async test functions appear to pass but never actually execute.

### Pitfall 4: KuzuDB segfault on garbage collection (Windows)

**What goes wrong:** Python segfaults when QueryResult objects are garbage collected after their parent Connection or Database is closed.
**Why it happens:** Use-after-free in KuzuDB's C++ bindings. Known issue #5457.
**How to avoid:** Always explicitly `del` QueryResult objects before closing Connection/Database. Call `gc.collect()` after deletion.
**Warning signs:** Random segfaults during test teardown or process exit.
**Phase impact:** Not Phase 01 (kuzu not installed yet), but document for Phase 03 memory layer.

### Pitfall 5: FastEmbed model download on first import

**What goes wrong:** First `TextEmbedding()` call downloads ~130MB model from HuggingFace. Slow. Can fail in offline environments.
**Why it happens:** FastEmbed downloads ONNX model files on first use.
**How to avoid:** Document this. In CI, pre-download the model. For the config verification test, mock the embedding model or use a lightweight check.
**Warning signs:** First test run takes 30-60 seconds. CI fails with network timeout.

**For Phase 01:** The success criterion says "Embedding model version is pinned in a config file and loadable at import time." This means the CONFIG must be loadable (the Pydantic model), not necessarily that the embedding model itself is loaded/downloaded at import. Verify the config is correct; defer actual model loading to Phase 03.

## Code Examples

### __init__.py with version from metadata

```python
# src/skill_retriever/__init__.py
"""Skill Retriever: Graph-based MCP server for Claude Code component retrieval."""

from importlib.metadata import version

__version__ = version("skill-retriever")
```

**Source:** [PEP 517](https://peps.python.org/pep-0517/), [importlib.metadata docs](https://docs.python.org/3/library/importlib.metadata.html)

**Why this works:** `uv_build` writes the version from `pyproject.toml` into the installed package metadata. `importlib.metadata.version()` reads it back. Single source of truth: the version in `pyproject.toml`.

### Complete pyproject.toml

```toml
[project]
name = "skill-retriever"
version = "0.1.0"
description = "Graph-based MCP server for Claude Code component retrieval"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastembed>=0.7.4",
    "pydantic>=2.12.5",
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "ruff>=0.14.14",
    "pyright>=1.1.408",
]

[build-system]
requires = ["uv_build>=0.9.28,<0.10.0"]
build-backend = "uv_build"

[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "W",    # pycodestyle warnings
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # ruff-specific rules
]

[tool.ruff.lint.isort]
known-first-party = ["skill_retriever"]

[tool.pyright]
pythonVersion = "3.13"
typeCheckingMode = "strict"
reportMissingTypeStubs = false

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Notes on ruff rules:**
- `TCH` (TYPE_CHECKING): Moves type-only imports behind `if TYPE_CHECKING`. Reduces runtime import cost.
- `RUF`: Ruff-specific rules catch Python anti-patterns.
- `N` (naming): Enforces PEP 8 naming conventions from day one.
- Line length 100: Readable without excessive wrapping. Matches STACK.md recommendation.

### Smoke test for Phase 01 success criteria

```python
# tests/test_init.py
"""Smoke tests for Phase 01 success criteria."""


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
```

### FastEmbed initialization (deferred to Phase 03, but showing the pattern)

```python
# This is NOT Phase 01 code — shown for reference
from fastembed import TextEmbedding

from skill_retriever.config import EMBEDDING_CONFIG

# Initialize with pinned config
embedding_model = TextEmbedding(
    model_name=EMBEDDING_CONFIG.model_name,
    max_length=EMBEDDING_CONFIG.max_length,
    cache_dir=EMBEDDING_CONFIG.cache_dir,
)

# Generate embeddings
embeddings = list(embedding_model.embed(["component description text"]))
# embeddings[0].shape == (384,)
```

**Source:** [FastEmbed GitHub](https://github.com/qdrant/fastembed), [FastEmbed PyPI](https://pypi.org/project/fastembed/)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| setup.py + pip | pyproject.toml + uv | 2024-2025 | Single config file, faster installs, lockfile |
| setuptools build | uv_build | Jan 2026 (uv 0.9) | Default for uv projects, fastest builds |
| Black + isort + Flake8 | Ruff (lint + format) | 2023-2024 | One tool, 10-100x faster |
| mypy | pyright | 2024-2025 | Faster, stricter by default, better IDE integration |
| `__version__ = "0.1.0"` in __init__.py | `importlib.metadata.version()` | PEP 517 (2017, widespread 2023+) | Single source of truth in pyproject.toml |
| requirements.txt | uv.lock | 2024-2025 | Deterministic, cross-platform lockfile |
| `--dev` flag for dev deps | `[dependency-groups]` (PEP 735) | 2025 | Standard way to declare dev dependencies in pyproject.toml |

**Deprecated/outdated:**
- setup.py: Replaced by pyproject.toml entirely
- MANIFEST.in: uv_build handles inclusion automatically
- tox.ini for pytest config: Use `[tool.pytest.ini_options]` in pyproject.toml

## Open Questions

1. **pydantic-settings vs plain BaseModel for config**
   - What we know: pydantic-settings adds env var override support. Plain BaseModel is simpler, zero extra deps.
   - What's unclear: Whether env var overrides are needed before Phase 06 (MCP server deployment).
   - Recommendation: Start with plain BaseModel. Add pydantic-settings in Phase 06 if env var config is needed. Avoid premature dependency.

2. **FastEmbed model pre-download for CI**
   - What we know: First `TextEmbedding()` call downloads ~130MB. CI may timeout.
   - What's unclear: Whether Phase 01 tests need to actually load the model or just verify the config.
   - Recommendation: Phase 01 tests verify config only (Pydantic model loads correctly, values match). Phase 03 tests will need the actual model. Add a CI step to pre-download then.

3. **uv dependency-groups vs --dev for dev dependencies**
   - What we know: PEP 735 `[dependency-groups]` is the modern standard. `uv add --dev` writes to `[dependency-groups] dev = [...]`.
   - What's unclear: Whether uv versions before 0.9 handle this correctly.
   - Recommendation: Use `uv add --dev` which auto-creates the `[dependency-groups]` section. Pin `uv>=0.9.28` in docs.

## Sources

### Primary (HIGH confidence)
- [uv docs — Creating projects](https://docs.astral.sh/uv/concepts/projects/init/) — `uv init --lib` structure, pyproject.toml defaults
- [uv docs — Build backend](https://docs.astral.sh/uv/concepts/build-backend/) — uv_build configuration, module discovery
- [pyright docs — Configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md) — pyproject.toml strict mode settings
- [FastEmbed GitHub](https://github.com/qdrant/fastembed) — TextEmbedding API, model initialization
- [FastEmbed PyPI](https://pypi.org/project/fastembed/) — v0.7.4, BAAI/bge-small-en-v1.5 default
- [pytest-asyncio docs](https://pytest-asyncio.readthedocs.io/en/latest/reference/configuration.html) — asyncio_mode = "auto"

### Secondary (MEDIUM confidence)
- [Ruff docs — Configuration](https://docs.astral.sh/ruff/configuration/) — lint rule selection
- [Python Developer Tooling Handbook](https://pydevtools.com/handbook/explanation/understanding-uv-init-project-types/) — uv project type comparison
- [KuzuDB Python API docs](https://docs.kuzudb.com/client-apis/python/) — API reference

### Tertiary (LOW confidence)
- [KuzuDB GitHub Issue #5457](https://github.com/kuzudb/kuzu/issues/5457) — Segfault on GC (Python bindings). Verified as known issue, but not tested on 0.11.3 specifically.
- [KuzuDB GitHub Issue #5801](https://github.com/kuzudb/kuzu/issues/5801) — Recursive relationship parameter bug on Windows 11 (reported on 0.11.0, may be fixed in 0.11.3).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified via PyPI and official docs
- Architecture: HIGH — uv init --lib is documented, Iusztin layers are per CLAUDE.md rules
- Pitfalls: MEDIUM — KuzuDB Windows issues are documented but not personally tested on 0.11.3

**Research date:** 2026-02-02
**Valid until:** 2026-03-04 (30 days — stable toolchain, no fast-moving components)
