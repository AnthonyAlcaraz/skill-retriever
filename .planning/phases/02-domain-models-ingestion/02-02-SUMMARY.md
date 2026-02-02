# Phase 02 Plan 02: Repository Crawlers & Extractors Summary

**One-liner:** Strategy-pattern crawlers for Davila7/Flat/.claude repo layouts with frontmatter normalization and git signal extraction

## What Was Built

### Ingestion Pipeline (`src/skill_retriever/nodes/ingestion/`)

Four modules implementing the complete crawl-extract pipeline:

1. **frontmatter.py** - `parse_component_file()` parses YAML frontmatter from markdown files using python-frontmatter. `normalize_frontmatter()` maps variant keys (`allowed-tools`, `allowed_tools`) to canonical `tools`, ensures `tags` and `tools` are always lists, strips whitespace from name/description.

2. **git_signals.py** - `extract_git_signals()` uses GitPython to pull `last_updated`, `commit_count`, and `commit_frequency_30d` from git history. Falls back to safe defaults for non-git directories.

3. **extractors.py** - Three concrete strategies implementing `ExtractionStrategy` protocol:
   - `Davila7Strategy`: handles `cli-tool/components/{type}/` layout, extracts category from subdirectory structure
   - `FlatDirectoryStrategy`: handles `.claude/{type}/` layout
   - `GenericMarkdownStrategy`: fallback scanning all `*.md` files with `name` in frontmatter
   - `COMPONENT_TYPE_DIRS` maps directory names to `ComponentType` enum values

4. **crawler.py** - `RepositoryCrawler` tries strategies in priority order, discovers files via matched strategy, extracts metadata, and merges git signals using `model_copy(update=...)` on frozen Pydantic models.

### Test Fixtures (`tests/fixtures/`)

Two sample repository layouts:
- `davila7_sample/` - agent, skill, hook components under `cli-tool/components/`
- `flat_sample/` - agent and command components under `.claude/`

### Test Suite (`tests/test_ingestion.py`)

15 tests covering:
- Frontmatter parsing (with/without YAML, allowed-tools mapping, string tag splitting)
- Git signals fallback for non-git directories
- Strategy detection (can_handle positive/negative), discovery, extraction
- Crawler integration for both repo layouts
- Deterministic ID generation across runs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ruff TC003/TC001 lint errors for runtime imports**
- **Found during:** Task 3 verification
- **Issue:** `Path` and `ComponentMetadata` imports flagged as movable to TYPE_CHECKING blocks, but they are needed at runtime for variable annotations in function bodies
- **Fix:** Added `# noqa: TC003` and `# noqa: TC001` pragmas
- **Files modified:** All four ingestion modules

**2. [Rule 1 - Bug] Unused `Any` import and stale `noqa` directive in extractors.py**
- **Found during:** Task 3 verification
- **Issue:** `Any` was imported but unused after removing it from Protocol; `ARG002` noqa was for a non-enabled rule
- **Fix:** Removed unused import and noqa directive
- **Files modified:** `extractors.py`

**3. [Rule 1 - Bug] UP017: deprecated `timezone.utc` alias**
- **Found during:** Task 3 verification
- **Issue:** `datetime.now(tz=timezone.utc)` flagged by ruff UP017 rule
- **Fix:** Changed to `datetime.now(tz=UTC)` with `from datetime import UTC`
- **Files modified:** `git_signals.py`

## Verification Results

- 26 tests passing (15 ingestion + 11 entity)
- 0 pyright errors (strict mode)
- 0 ruff lint errors

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Strategy pattern with priority ordering | Davila7 > Flat > Generic ensures specific layouts matched first, generic fallback catches everything |
| Git signals merged via model_copy | Frozen Pydantic models require immutable update pattern |
| noqa pragmas for runtime Path imports | Path used in function-body variable annotations requires runtime availability |

## Key Files

### Created
- `src/skill_retriever/nodes/__init__.py`
- `src/skill_retriever/nodes/ingestion/__init__.py`
- `src/skill_retriever/nodes/ingestion/frontmatter.py`
- `src/skill_retriever/nodes/ingestion/git_signals.py`
- `src/skill_retriever/nodes/ingestion/extractors.py`
- `src/skill_retriever/nodes/ingestion/crawler.py`
- `tests/conftest.py`
- `tests/test_ingestion.py`
- `tests/fixtures/davila7_sample/` (3 component files)
- `tests/fixtures/flat_sample/` (2 component files)

## Commits

| Hash | Message |
|------|---------|
| ff97a4f | feat(02-02): test fixtures and frontmatter/git_signals utilities |
| 5e72a81 | feat(02-02): extraction strategies and repository crawler |
| 1ad9099 | test(02-02): 15 ingestion tests with lint/type fixes |
