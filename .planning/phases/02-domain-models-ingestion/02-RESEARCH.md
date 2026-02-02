# Phase 02: Domain Models & Ingestion - Research

**Researched:** 2026-02-02
**Domain:** Pydantic entity modeling, Git repository crawling, entity resolution/deduplication
**Confidence:** HIGH

## Summary

This phase builds three layers: (1) Pydantic v2 domain models for the 7 component types in Claude Code ecosystems, (2) a strategy-pattern repository crawler that extracts components from any repo structure, and (3) a two-phase entity resolution pipeline combining fuzzy string matching with embedding similarity.

The davila7/claude-code-templates repository serves as the reference implementation but the crawler must handle arbitrary structures. Component definitions are markdown files with YAML frontmatter containing `name`, `description`, `tools`/`allowed-tools`, `version`, and `model`/`priority` fields. The repository organizes components under `cli-tool/components/{type}/{category}/{component-name}/` with individual `.md` files or `SKILL.md` files inside named directories.

GitPython provides the simplest path to extracting git health signals (last commit date, commit frequency) without requiring C library dependencies. RapidFuzz is the clear choice over TheFuzz for fuzzy string matching in the entity resolution pipeline, offering C++ performance with API compatibility. FastEmbed (already a project dependency) handles the embedding similarity phase.

**Primary recommendation:** Build thin Pydantic models with Literal-discriminated unions for component types, use Protocol-based strategy pattern for extractors, and keep entity resolution as a standalone pipeline stage that runs after all extraction completes.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | >=2.12.5 | Entity models, validation, serialization | Already pinned in pyproject.toml; Rust-backed validation, discriminated unions |
| gitpython | >=3.1.44 | Git metadata extraction (commit dates, frequency) | Pure Python, no C deps, wraps git CLI, mature (3.1.x stable) |
| rapidfuzz | >=3.14 | Fuzzy string matching for entity resolution | C++ performance, MIT license, drop-in TheFuzz replacement |
| fastembed | >=0.7.4 | Embedding similarity for entity resolution phase 2 | Already pinned in pyproject.toml; BAAI/bge-small-en-v1.5 configured |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyyaml | >=6.0 | Parse YAML frontmatter from markdown component files | Every component file extraction |
| python-frontmatter | >=1.1 | Higher-level frontmatter parsing (wraps pyyaml) | Cleaner API for markdown+YAML files |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| gitpython | pygit2 | pygit2 is faster (libgit2 C bindings) but requires compiled libgit2 dependency; gitpython is simpler for our needs (commit metadata only, not heavy git operations) |
| gitpython | subprocess git | Lower-level, more parsing work; gitpython already wraps this cleanly |
| rapidfuzz | thefuzz | TheFuzz is slower (pure Python Levenshtein), no longer actively developed; RapidFuzz is the community successor |
| python-frontmatter | manual yaml.safe_load | python-frontmatter handles edge cases (no frontmatter, malformed, content split) that manual parsing misses |

**Installation:**
```bash
uv add gitpython rapidfuzz python-frontmatter
```

## Architecture Patterns

### Recommended Project Structure
```
src/skill_retriever/
├── entities/
│   ├── __init__.py           # Re-exports all models
│   ├── components.py         # ComponentMetadata, ComponentType enum
│   ├── capabilities.py       # CapabilityEntity
│   └── graph.py              # GraphNode, GraphEdge, EdgeType
├── nodes/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── crawler.py        # RepositoryCrawler (discovers files)
│   │   ├── extractors.py     # ExtractionStrategy Protocol + implementations
│   │   ├── frontmatter.py    # Markdown/YAML parsing utilities
│   │   ├── git_signals.py    # GitPython health signal extraction
│   │   └── resolver.py       # Entity resolution pipeline
│   └── ...
```

### Pattern 1: Literal-Discriminated Component Types
**What:** Use Pydantic's Literal discriminator to distinguish 7 component types in a union
**When to use:** Serializing/deserializing mixed component collections; validating component type at parse time
**Example:**
```python
from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Literal

class ComponentType(StrEnum):
    AGENT = "agent"
    SKILL = "skill"
    COMMAND = "command"
    SETTING = "setting"
    MCP = "mcp"
    HOOK = "hook"
    SANDBOX = "sandbox"

class ComponentMetadata(BaseModel):
    """Core entity representing any Claude Code component."""
    id: str = Field(description="Deterministic ID: {repo_owner}/{repo_name}/{type}/{name}")
    name: str
    component_type: ComponentType
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    version: str = ""

    # Git health signals (INGS-04)
    last_updated: datetime | None = None
    commit_count: int = 0
    commit_frequency_30d: float = 0.0  # commits per day in last 30 days

    # Full definition (INGS-03)
    raw_content: str = ""  # Full markdown content
    parameters: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    # Source tracking
    source_repo: str = ""
    source_path: str = ""  # Relative path within repo
    category: str = ""     # e.g., "ai-specialists", "development"
```
**Source:** Pydantic v2 discriminated unions documentation (https://docs.pydantic.dev/latest/concepts/unions/)

### Pattern 2: Protocol-Based Extraction Strategy
**What:** Use `typing.Protocol` to define extraction interface; each repo structure gets its own concrete strategy
**When to use:** Supporting multiple repository layouts without conditional branching in the crawler
**Example:**
```python
from typing import Protocol, runtime_checkable
from pathlib import Path

@runtime_checkable
class ExtractionStrategy(Protocol):
    """Strategy for extracting components from a specific repo structure."""

    def can_handle(self, repo_root: Path) -> bool:
        """Return True if this strategy understands the repo layout."""
        ...

    def discover(self, repo_root: Path) -> list[Path]:
        """Return paths to all component definition files."""
        ...

    def extract(self, file_path: Path, repo_root: Path) -> ComponentMetadata:
        """Parse a single component file into a ComponentMetadata entity."""
        ...

class Davila7Strategy:
    """Extracts from cli-tool/components/{type}/{category}/{name}/ structure."""

    def can_handle(self, repo_root: Path) -> bool:
        return (repo_root / "cli-tool" / "components").is_dir()

    def discover(self, repo_root: Path) -> list[Path]:
        components_dir = repo_root / "cli-tool" / "components"
        paths = []
        for type_dir in components_dir.iterdir():
            if type_dir.is_dir() and type_dir.name in COMPONENT_TYPE_DIRS:
                for md_file in type_dir.rglob("*.md"):
                    paths.append(md_file)
        return paths

class FlatDirectoryStrategy:
    """Extracts from flat .claude/ directory structure."""

    def can_handle(self, repo_root: Path) -> bool:
        return (repo_root / ".claude").is_dir()
```
**Source:** PEP 544 Protocol classes, Strategy pattern (https://refactoring.guru/design-patterns/strategy/python/example)

### Pattern 3: Two-Phase Entity Resolution
**What:** First pass uses fuzzy string matching (RapidFuzz) on names/descriptions to find candidates; second pass computes embedding similarity to confirm matches above 0.85 threshold
**When to use:** After all extraction completes, before graph insertion
**Example:**
```python
from rapidfuzz import fuzz, process
from fastembed import TextEmbedding
import numpy as np

class EntityResolver:
    def __init__(self, embedding_model: TextEmbedding,
                 fuzzy_threshold: float = 80.0,
                 embedding_threshold: float = 0.85):
        self.embedding_model = embedding_model
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold

    def resolve(self, entities: list[ComponentMetadata]) -> list[ComponentMetadata]:
        """Deduplicate entities using two-phase resolution."""
        # Phase 1: Fuzzy string matching on names
        candidates = self._find_fuzzy_candidates(entities)

        # Phase 2: Embedding similarity confirmation
        confirmed_duplicates = self._confirm_with_embeddings(candidates)

        # Merge duplicates, keeping richest metadata
        return self._merge_duplicates(entities, confirmed_duplicates)

    def _find_fuzzy_candidates(self, entities):
        names = [e.name for e in entities]
        pairs = []
        for i, name in enumerate(names):
            matches = process.extract(name, names[i+1:], scorer=fuzz.token_sort_ratio,
                                       score_cutoff=self.fuzzy_threshold)
            for match_name, score, idx in matches:
                pairs.append((i, i + 1 + idx, score))
        return pairs
```

### Pattern 4: Deterministic Component IDs
**What:** Generate stable, reproducible IDs from source location: `{repo_owner}/{repo_name}/{type}/{normalized_name}`
**When to use:** Every component gets an ID at extraction time; IDs must be stable across re-ingestion runs
**Why:** Enables incremental updates and cross-repo deduplication without UUIDs

### Anti-Patterns to Avoid
- **Parsing markdown with regex:** Use python-frontmatter for YAML extraction; regex fails on edge cases (nested code blocks, multiline values, no frontmatter)
- **Loading entire repo history into memory:** Use `iter_commits(paths=file, max_count=N)` with GitPython; never `list(repo.iter_commits())` on large repos
- **Tight coupling between crawler and extractor:** Crawler discovers files, strategy extracts metadata. Never put repo-structure-specific logic in the crawler
- **Mutable entity models:** ComponentMetadata should be functionally immutable after creation. Use `model_copy(update={...})` for modifications
- **Running entity resolution during extraction:** Resolution must be a separate pipeline stage after ALL extraction completes, because it needs the full entity set for comparison

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML frontmatter parsing | Regex-based frontmatter extraction | python-frontmatter | Handles missing frontmatter, malformed YAML, content splitting, encoding edge cases |
| Fuzzy string matching | Custom Levenshtein implementation | rapidfuzz | C++ performance, battle-tested scorers (token_sort_ratio, partial_ratio), `process.extract` for bulk matching |
| Git commit metadata | subprocess + git log parsing | gitpython `Repo.iter_commits()` | Object model for commits, handles encoding, cross-platform |
| Embedding computation | Manual model loading + tokenization | fastembed (already in deps) | Handles batching, ONNX runtime, model caching |
| Component ID generation | UUID or random strings | Deterministic path-based IDs | Enables incremental re-ingestion, stable references, cross-session consistency |
| Markdown content extraction | BeautifulSoup or custom parser | python-frontmatter `.content` attribute | Cleanly separates frontmatter from body; body is already the "full definition" |

**Key insight:** The davila7 repo components are markdown files with YAML frontmatter. This is a solved problem in the Python ecosystem. The complexity is in handling the variety of repo structures and the entity resolution, not the parsing itself.

## Common Pitfalls

### Pitfall 1: Assuming Uniform Frontmatter Fields Across Component Types
**What goes wrong:** Agent files have `name`, `description`, `tools`, `model`. Skill files have `name`, `description`, `allowed-tools`, `version`, `priority`. Hook files are JSON-based (`HOOK_PATTERNS_COMPRESSED.json`), not markdown at all. Settings and sandbox components may have entirely different structures.
**Why it happens:** Testing only against agent files during development
**How to avoid:** Define a minimal required field set (name, description) and treat everything else as optional. Use `model_validator(mode='before')` to normalize field names across formats (e.g., `tools` and `allowed-tools` both map to `tools` field).
**Warning signs:** Tests pass for agents but fail for skills or hooks

### Pitfall 2: Git Operations on Cloned Repos Without .git Directory
**What goes wrong:** If repo was downloaded as ZIP or tarball (no `.git`), GitPython throws `InvalidGitRepositoryError`
**Why it happens:** Not all users will `git clone`; some may download and extract
**How to avoid:** Check for `.git` directory before attempting git operations. When absent, skip git health signals gracefully (set defaults: `last_updated=None`, `commit_count=0`)
**Warning signs:** `InvalidGitRepositoryError` in tests using fixture directories

### Pitfall 3: Entity Resolution Quadratic Blowup
**What goes wrong:** Comparing every entity against every other entity is O(n^2). With ~1400 components, that is ~1M comparisons for fuzzy matching alone, plus embedding computation
**Why it happens:** Naive all-pairs comparison without blocking
**How to avoid:** Use blocking strategy: only compare entities of the same `component_type`. This reduces from ~1400^2 to sum of (type_count^2), roughly 10x reduction. Also use `rapidfuzz.process.extract` with `score_cutoff` to skip low-scoring pairs early
**Warning signs:** Entity resolution takes >10 seconds on the full dataset

### Pitfall 4: Confusing File Structure with Component Type
**What goes wrong:** In davila7, the directory name maps to component type (`agents/` = agent, `skills/` = skill). In flat repos, there may be no such directory structure. Mapping logic that depends on directory names breaks for other repos.
**How to avoid:** File path hints are one signal; frontmatter content is another. The ExtractionStrategy is responsible for determining component type, not the crawler. For flat repos, infer type from frontmatter fields or file content patterns.
**Warning signs:** All components from a flat repo get classified as "unknown" type

### Pitfall 5: Over-Indexing on davila7 Structure
**What goes wrong:** Building an extractor that perfectly handles `cli-tool/components/{type}/{category}/{name}/` but fails on any other layout
**Why it happens:** Only testing against one repo
**How to avoid:** Success criteria #2 explicitly requires a flat-directory test. Create a minimal test fixture with 3-5 components in a flat `.claude/` structure during plan 02-02
**Warning signs:** Strategy only has `Davila7Strategy`, no `FlatDirectoryStrategy` or `GenericMarkdownStrategy`

### Pitfall 6: Forgetting the marketplace.json Metadata Source
**What goes wrong:** Ignoring `marketplace.json` means missing bundle-level metadata (keywords, version, license, mcpServers) that enriches individual components
**Why it happens:** Focus on individual .md files, not aggregate metadata files
**How to avoid:** Davila7Strategy should parse `marketplace.json` first to build a keyword/bundle index, then enrich individual components during extraction
**Warning signs:** Components have no tags/keywords despite marketplace.json containing rich keyword arrays

## Code Examples

### Frontmatter Parsing with python-frontmatter
```python
import frontmatter

def parse_component_file(file_path: Path) -> tuple[dict, str]:
    """Parse a component markdown file into metadata dict and content body."""
    post = frontmatter.load(str(file_path))
    metadata = dict(post.metadata)  # YAML frontmatter as dict
    content = post.content           # Markdown body
    return metadata, content

# Agent example yields:
# metadata = {"name": "prompt-engineer", "description": "...", "tools": ["Read", "Write", "Edit"], "model": "opus"}
# content = "## Expertise Areas\n..."
```

### Git Health Signal Extraction with GitPython
```python
from git import Repo, InvalidGitRepositoryError
from datetime import datetime, timedelta, timezone

def extract_git_signals(repo_path: Path, file_relative_path: str) -> dict:
    """Extract git health signals for a specific file."""
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        return {"last_updated": None, "commit_count": 0, "commit_frequency_30d": 0.0}

    commits = list(repo.iter_commits(paths=file_relative_path, max_count=500))
    if not commits:
        return {"last_updated": None, "commit_count": 0, "commit_frequency_30d": 0.0}

    last_updated = commits[0].committed_datetime
    commit_count = len(commits)

    # Commits in last 30 days
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    recent = [c for c in commits if c.committed_datetime > thirty_days_ago]
    frequency_30d = len(recent) / 30.0

    return {
        "last_updated": last_updated,
        "commit_count": commit_count,
        "commit_frequency_30d": round(frequency_30d, 4),
    }
```

### RapidFuzz Entity Matching
```python
from rapidfuzz import fuzz, process

def find_fuzzy_duplicates(
    names: list[str],
    threshold: float = 80.0
) -> list[tuple[int, int, float]]:
    """Find name pairs above fuzzy similarity threshold."""
    duplicates = []
    for i in range(len(names)):
        # Only compare forward to avoid duplicate pairs
        remaining = names[i + 1:]
        if not remaining:
            continue
        matches = process.extract(
            names[i],
            remaining,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        for match_name, score, idx in matches:
            duplicates.append((i, i + 1 + idx, score))
    return duplicates
```

### Embedding Similarity Confirmation
```python
from fastembed import TextEmbedding
import numpy as np

def confirm_duplicates_with_embeddings(
    entities: list[ComponentMetadata],
    candidate_pairs: list[tuple[int, int, float]],
    model: TextEmbedding,
    threshold: float = 0.85,
) -> list[tuple[int, int]]:
    """Confirm fuzzy candidates using embedding cosine similarity."""
    if not candidate_pairs:
        return []

    # Batch embed all unique indices
    unique_indices = set()
    for i, j, _ in candidate_pairs:
        unique_indices.add(i)
        unique_indices.add(j)

    texts = [f"{entities[i].name} {entities[i].description}" for i in sorted(unique_indices)]
    embeddings = list(model.embed(texts))
    idx_to_emb = dict(zip(sorted(unique_indices), embeddings))

    confirmed = []
    for i, j, fuzzy_score in candidate_pairs:
        emb_i = idx_to_emb[i]
        emb_j = idx_to_emb[j]
        cosine_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
        if cosine_sim >= threshold:
            confirmed.append((i, j))
    return confirmed
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| fuzzywuzzy (GPL) | rapidfuzz (MIT, C++) | 2022+ | 10-50x faster, permissive license, active maintenance |
| Pydantic v1 validators | Pydantic v2 `field_validator` / `model_validator` | 2023 (v2.0) | Rust-backed core, discriminated unions in Rust, `model_copy` replaces `.copy()` |
| Manual YAML parsing | python-frontmatter | Stable since 2018 | Handles all edge cases of markdown+YAML |
| pygit2 for simple git ops | gitpython for metadata-only use cases | Ongoing | gitpython is simpler when you only need commit iteration, no C deps |

**Deprecated/outdated:**
- `fuzzywuzzy`: Replaced by `thefuzz` (rename) and superseded by `rapidfuzz` (performance)
- Pydantic v1 `@validator` decorator: Replaced by `@field_validator` in v2
- Pydantic v1 `.copy(update={})`: Replaced by `.model_copy(update={})` in v2

## Open Questions

1. **Hook component format**
   - What we know: Hooks use `HOOK_PATTERNS_COMPRESSED.json` (JSON), not markdown. Individual hook dirs contain markdown files but the compressed JSON is the aggregate.
   - What's unclear: Do individual hook `.md` files have frontmatter, or are they pure instruction text? The compressed JSON suggests hooks may need a JSON-based extractor alongside the markdown one.
   - Recommendation: Build a `HookJsonExtractor` alongside the markdown-based strategies. Parse both the compressed JSON and individual `.md` files to get maximum coverage.

2. **marketplace.json as metadata enrichment source**
   - What we know: 10 plugin bundles in marketplace.json with keywords, version, license, mcpServers arrays. Components referenced by file path.
   - What's unclear: How to merge bundle-level keywords with individual component metadata. Should bundle keywords propagate to all components in that bundle?
   - Recommendation: Parse marketplace.json early. Propagate bundle keywords as tags on each component in that bundle. Store bundle membership as metadata for potential graph edges later.

3. **Flat repo structure detection heuristic**
   - What we know: Flat repos have `.claude/agents/` and `.claude/commands/` directly. No `cli-tool/components/` hierarchy.
   - What's unclear: How many real-world repos use flat vs hierarchical structures? Are there other common patterns?
   - Recommendation: Build `Davila7Strategy` and `FlatClaudeStrategy` for v1. Add a `GenericMarkdownStrategy` that recursively scans for any `.md` files with recognized frontmatter fields as fallback.

4. **Incremental re-ingestion scope**
   - What we know: Context mentions "incremental updates after initial full build"
   - What's unclear: Whether Phase 2 needs to support incremental, or if full rebuild is sufficient for v1
   - Recommendation: Phase 2 builds full-rebuild only. Use deterministic IDs so that Phase 3 (memory layer) can detect "already ingested" components by ID. Incremental is a v2 optimization.

## Reference: davila7/claude-code-templates Repository Structure

```
davila7/claude-code-templates/
├── .claude/
│   ├── agents/                    # Flat structure (empty in this repo)
│   └── commands/                  # Flat structure (empty in this repo)
├── .claude-plugin/
│   └── marketplace.json           # 10 bundles with keywords, versions, component paths
├── cli-tool/
│   └── components/
│       ├── agents/                # 27+ category dirs, each with .md files
│       │   ├── ai-specialists/    # 7 agents (e.g., prompt-engineer.md)
│       │   ├── database/
│       │   ├── development-team/
│       │   └── ...
│       ├── skills/                # 18 category dirs, each with {skill-name}/SKILL.md
│       │   ├── ai-research/       # ~30 skills
│       │   ├── development/       # clean-code/, architecture/, etc.
│       │   └── ...
│       ├── commands/              # 22 category dirs with command .md files
│       ├── hooks/                 # 10 category dirs + HOOK_PATTERNS_COMPRESSED.json
│       ├── settings/              # 12 category dirs
│       ├── mcps/                  # 11 category dirs
│       └── sandbox/               # 3 dirs (cloudflare, docker, e2b)
```

**Component file formats observed:**
- Agents: `{category}/{agent-name}.md` with YAML frontmatter (`name`, `description`, `tools`, `model`)
- Skills: `{category}/{skill-name}/SKILL.md` with YAML frontmatter (`name`, `description`, `allowed-tools`, `version`, `priority`)
- Commands: `{category}/{command-name}.md` (format TBD, likely similar frontmatter)
- Hooks: JSON (`HOOK_PATTERNS_COMPRESSED.json`) + individual `.md` files in category dirs
- Settings: category dirs with configuration files (format TBD)
- MCPs: category dirs with MCP definition files (format TBD)
- Sandbox: `README.md` + implementation dirs (cloudflare, docker, e2b)

## Sources

### Primary (HIGH confidence)
- davila7/claude-code-templates GitHub API - repository tree, file contents, marketplace.json schema verified directly
- Pydantic v2 official docs (https://docs.pydantic.dev/latest/concepts/unions/) - discriminated unions, validators
- Pydantic v2 validators docs (https://docs.pydantic.dev/latest/concepts/validators/) - field_validator, model_validator

### Secondary (MEDIUM confidence)
- GitPython documentation (https://gitpython.readthedocs.io/en/stable/tutorial.html) - iter_commits, commit metadata APIs
- RapidFuzz documentation (https://rapidfuzz.github.io/RapidFuzz/) - process.extract, scorer options, score_cutoff
- RapidFuzz GitHub (https://github.com/rapidfuzz/RapidFuzz) - version 3.14.x, C++ backend confirmed
- python-frontmatter PyPI (https://pypi.org/project/python-frontmatter/) - frontmatter.load API
- PEP 544 Protocol classes - typing.Protocol for strategy pattern

### Tertiary (LOW confidence)
- Component counts (~329 agents, ~664 skills, etc.) from architecture research context - not independently verified against current repo state

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries verified via official docs/repos, versions confirmed
- Architecture: HIGH - patterns verified against actual davila7 repo structure, Protocol strategy pattern is well-established Python idiom
- Pitfalls: MEDIUM - derived from structural analysis of the repo and general engineering experience; some pitfalls (quadratic blowup thresholds) would benefit from empirical validation

**Research date:** 2026-02-02
**Valid until:** 2026-03-04 (30 days - stable domain, libraries are mature)
