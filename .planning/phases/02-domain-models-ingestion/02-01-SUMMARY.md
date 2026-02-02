# Plan 02-01 Summary: Pydantic Entity Models

## What was done

1. **Fixed TYPE_CHECKING guards** — `datetime` in `components.py` and `ComponentType` in `graph.py` were behind `TYPE_CHECKING` blocks, which breaks Pydantic runtime model resolution when `from __future__ import annotations` is active. Moved to runtime imports with `# noqa: TC003/TC001` to suppress ruff's TCH rules (correct: Pydantic needs these at runtime).

2. **Dependencies verified** — `gitpython>=3.1.44`, `rapidfuzz>=3.14`, `python-frontmatter>=1.1` all importable (were already in pyproject.toml from Phase 1).

3. **Created `tests/test_entities.py`** — 11 test cases covering:
   - ComponentType enum values (7 types, all lowercase strings)
   - ComponentMetadata creation with all fields + round-trip serialization
   - Default values for optional fields
   - `generate_id()` deterministic ID generation
   - Special character normalization in IDs (spaces, mixed case, whitespace)
   - Frozen model immutability (raises ValidationError on mutation)
   - `model_copy(update={})` produces new instance without modifying original
   - ID field validator normalization
   - GraphNode creation
   - GraphEdge creation with EdgeType
   - EdgeType enum values (5 types)

## Verification results

- **pytest**: 11/11 passed (0.22s)
- **pyright strict**: 0 errors, 0 warnings
- **ruff**: All checks passed
- **Integration**: All 5 entity types importable from `skill_retriever.entities`

## Files modified

| File | Change |
|------|--------|
| `src/skill_retriever/entities/components.py` | Moved `datetime` import out of TYPE_CHECKING |
| `src/skill_retriever/entities/graph.py` | Moved `ComponentType` import out of TYPE_CHECKING |
| `tests/test_entities.py` | **New** — 11 entity model tests |

## Decisions

- Used `# noqa: TC003/TC001` rather than `model_rebuild()` — simpler, keeps imports explicit, and correctly documents that Pydantic needs these at runtime despite `from __future__ import annotations`.
