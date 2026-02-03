# Phase 6: MCP Server & Installation - Research

**Researched:** 2026-02-03
**Domain:** MCP Protocol, FastMCP, File Installation, Rationale Generation
**Confidence:** HIGH

## Summary

Phase 6 exposes the skill-retriever system as an MCP server callable from Claude Code and implements component installation into the `.claude/` directory structure. Research confirms FastMCP 2.x is the standard framework for Python MCP servers, providing automatic schema generation from type hints and docstrings. The MCP protocol is now a Linux Foundation standard adopted by OpenAI, Google, and major AI tooling.

Key constraints: tool schemas must stay under 300 tokens total (5 tools), and each recommendation must include graph-path rationale explaining why it was selected. The existing `RetrievalPath` dataclass in `flow_pruner.py` already captures the path structure needed for rationale generation.

**Primary recommendation:** Use FastMCP 2.x with minimal tool descriptions, leverage Pydantic models for input/output schemas, and generate rationale from the existing `RetrievalPath.nodes` list by converting graph edges to human-readable explanations.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fastmcp | 2.14.4 | MCP server framework | Official Python MCP framework, 1M+ daily downloads, auto-generates tool schemas |
| pydantic | >=2.12 | Input/output models | Already in project, type-safe schema generation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| shutil | stdlib | File operations | Component installation to .claude/ |
| pathlib | stdlib | Path manipulation | Cross-platform directory handling |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastMCP | mcp (official SDK) | FastMCP wraps official SDK with simpler decorator API; use FastMCP |
| shutil.copy | manual file.write | shutil handles permissions, metadata; use shutil |

**Installation:**
```bash
uv add "fastmcp<3"
```

Pin to v2 explicitly. FastMCP 3.0 is in beta and may introduce breaking changes.

## Architecture Patterns

### Recommended Project Structure
```
src/skill_retriever/
├── mcp/                    # Serving layer (Iusztin)
│   ├── __init__.py
│   ├── server.py           # FastMCP server with tool definitions
│   ├── tools.py            # Tool handler implementations
│   ├── schemas.py          # Pydantic input/output models
│   ├── rationale.py        # Path-to-explanation converter
│   └── installer.py        # Component installation logic
```

### Pattern 1: Minimal Tool Schema Design
**What:** Keep tool descriptions under 60 tokens each to stay within 300-token budget for 5 tools.
**When to use:** Always for MCP servers consumed by Claude Code.
**Example:**
```python
# Source: https://modelcontextprotocol.io/docs/develop/build-server
from fastmcp import FastMCP

mcp = FastMCP("skill-retriever")

@mcp.tool
async def search_components(query: str, top_k: int = 5) -> SearchResult:
    """Search components by task description."""  # Keep docstring under 10 words
    ...
```

### Pattern 2: Pydantic Models for Structured Responses
**What:** Define Pydantic models for tool inputs and outputs. FastMCP auto-generates JSON Schema.
**When to use:** All tools with complex inputs or outputs.
**Example:**
```python
# Source: FastMCP documentation
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    components: list[ComponentRecommendation]
    total_tokens: int = Field(description="Estimated context cost")
    conflicts: list[str] = []
```

### Pattern 3: Path-Based Rationale Generation
**What:** Convert graph traversal paths to natural language explanations.
**When to use:** For INTG-03 (graph-path rationale).
**Example:**
```python
# Convert RetrievalPath.nodes to rationale
def generate_rationale(paths: list[RetrievalPath], graph_store: GraphStore) -> str:
    """Generate explanation from graph paths."""
    explanations = []
    for path in paths[:3]:  # Top 3 paths
        # path.nodes = ["query-node", "intermediate", "target-component"]
        edge_descriptions = []
        for i in range(len(path.nodes) - 1):
            edge = get_edge_between(path.nodes[i], path.nodes[i+1], graph_store)
            edge_descriptions.append(f"{edge.edge_type.value}")
        explanations.append(" -> ".join(edge_descriptions))
    return "; ".join(explanations)
```

### Pattern 4: .claude/ Directory Installation
**What:** Install components to the correct subdirectory based on ComponentType.
**When to use:** install_components tool implementation.
**Example:**
```python
# Map ComponentType to .claude/ subdirectory
INSTALL_PATHS = {
    ComponentType.SKILL: ".claude/skills/{name}/SKILL.md",
    ComponentType.COMMAND: ".claude/commands/{name}.md",
    ComponentType.AGENT: ".claude/agents/{name}.md",
    ComponentType.SETTING: ".claude/settings.json",  # Merge
    ComponentType.HOOK: ".claude/hooks/{name}/",
    ComponentType.MCP: ".claude/mcp-servers/{name}/",
    ComponentType.SANDBOX: ".claude/sandbox/{name}/",
}
```

### Anti-Patterns to Avoid
- **Verbose tool descriptions:** Each word costs tokens. "Search components by natural language task description and return ranked recommendations with dependencies" is 12 words. Use "Search components by task description" (5 words).
- **Flat component installation:** Don't dump all components into a single directory. Follow the .claude/ hierarchy.
- **Synchronous MCP handlers:** FastMCP supports async; use async for I/O operations.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON Schema generation | Custom schema builder | Pydantic + FastMCP | FastMCP auto-generates from type hints |
| MCP protocol handling | Custom JSON-RPC | FastMCP | Handles transport, validation, errors |
| File copying with permissions | open/write loop | shutil.copy2 | Preserves metadata, handles errors |
| Directory creation | os.makedirs | pathlib.mkdir(parents=True) | Cross-platform, cleaner API |
| Token estimation | Character counting | Use existing estimate_tokens() | Already in context_assembler.py |

**Key insight:** FastMCP's automatic schema generation from docstrings means the tool definition IS the documentation. Optimize docstrings for token efficiency, not human readability.

## Common Pitfalls

### Pitfall 1: Tool Schema Token Explosion
**What goes wrong:** Tool schemas exceed 300 tokens, consuming valuable context window.
**Why it happens:** Verbose descriptions, nested object schemas with full descriptions.
**How to avoid:**
- Measure schema tokens before shipping
- Use simple types (str, int, list[str]) where possible
- Descriptions under 10 words per field
**Warning signs:** `/context` command shows MCP server consuming >5% of context.

### Pitfall 2: STDIO Logging to STDOUT
**What goes wrong:** Server crashes or produces malformed responses.
**Why it happens:** print() statements corrupt JSON-RPC messages on STDIO transport.
**How to avoid:**
- Use logging module with stderr handler
- Never use print() in MCP server code
- Configure logging in server initialization
**Warning signs:** Server works in tests but fails when connected to Claude Code.

### Pitfall 3: Missing Dependency Resolution Before Installation
**What goes wrong:** User installs component A, but it requires component B which isn't installed.
**Why it happens:** install_components doesn't call resolve_transitive_dependencies first.
**How to avoid:**
- Always run dependency resolution before installation
- Return list of components that WILL be installed (including deps)
- Warn on conflicts before installation
**Warning signs:** Installed components reference missing skills/MCPs.

### Pitfall 4: Blocking Installation Operations
**What goes wrong:** MCP server times out during installation.
**Why it happens:** File I/O is synchronous in async handler.
**How to avoid:**
- Use asyncio.to_thread() for file operations
- Or accept that installation is fast enough (<100ms) for sync
**Warning signs:** Installation of multiple components causes timeout.

### Pitfall 5: Settings.json Merge Conflicts
**What goes wrong:** Installing a setting overwrites user's existing settings.
**Why it happens:** Naive file replacement instead of deep merge.
**How to avoid:**
- Load existing settings.json
- Deep merge new settings
- Write merged result
**Warning signs:** User reports lost settings after component installation.

## Code Examples

Verified patterns from official sources:

### FastMCP Tool Definition
```python
# Source: https://github.com/jlowin/fastmcp
from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("skill-retriever")

class SearchInput(BaseModel):
    query: str
    top_k: int = 5
    component_type: str | None = None

class ComponentRec(BaseModel):
    id: str
    name: str
    score: float
    rationale: str
    token_cost: int

class SearchResult(BaseModel):
    components: list[ComponentRec]
    total_tokens: int
    conflicts: list[str]

@mcp.tool
async def search_components(input: SearchInput) -> SearchResult:
    """Search components by task."""
    # Implementation calls RetrievalPipeline.retrieve()
    ...
```

### Server Entry Point
```python
# Source: https://modelcontextprotocol.io/docs/develop/build-server
def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

### Path-to-Rationale Conversion
```python
# Pattern derived from existing flow_pruner.py
from skill_retriever.entities.graph import EdgeType

EDGE_DESCRIPTIONS = {
    EdgeType.DEPENDS_ON: "requires",
    EdgeType.ENHANCES: "enhances",
    EdgeType.CONFLICTS_WITH: "conflicts with",
}

def path_to_rationale(path_nodes: list[str], graph_store: GraphStore) -> str:
    """Convert graph path to human-readable rationale."""
    if len(path_nodes) < 2:
        return "Direct match"

    parts = []
    for i in range(len(path_nodes) - 1):
        edges = graph_store.get_edges(path_nodes[i])
        for edge in edges:
            if edge.target_id == path_nodes[i + 1]:
                desc = EDGE_DESCRIPTIONS.get(edge.edge_type, "relates to")
                parts.append(f"{path_nodes[i]} {desc} {path_nodes[i + 1]}")
                break
    return " -> ".join(parts) if parts else "Graph traversal"
```

### Component Installation
```python
# Pattern for .claude/ installation
from pathlib import Path
import shutil

def install_component(
    component: ComponentMetadata,
    target_dir: Path,
) -> Path:
    """Install a component to the correct .claude/ subdirectory."""
    # Determine target path based on type
    if component.component_type == ComponentType.SKILL:
        dest = target_dir / ".claude" / "skills" / component.name / "SKILL.md"
    elif component.component_type == ComponentType.COMMAND:
        dest = target_dir / ".claude" / "commands" / f"{component.name}.md"
    elif component.component_type == ComponentType.AGENT:
        dest = target_dir / ".claude" / "agents" / f"{component.name}.md"
    # ... other types

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(component.raw_content, encoding="utf-8")
    return dest
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| mcp SDK directly | FastMCP wrapper | Q1 2025 | Simpler API, auto-schema |
| .claude/commands/ only | .claude/skills/ with SKILL.md | Jan 2026 | Skills subsume commands |
| Separate command/skill systems | Unified skill system | Jan 10, 2026 | Single mental model |
| JSON tool schemas | Pydantic auto-generation | FastMCP 2.0 | Type-safe, less boilerplate |

**Deprecated/outdated:**
- `mcp[cli]` package: Still works but FastMCP recommended for new servers
- `.claude/commands/` as primary: Skills with SKILL.md are now preferred
- FastMCP 1.x: Incorporated into official SDK; use FastMCP 2.x standalone

## Tool Schema Token Budget

Per success criteria, total tool schema definitions must stay under 300 tokens.

**Token estimates for 5 tools:**

| Tool | Description | Input Fields | Est. Tokens |
|------|-------------|--------------|-------------|
| search_components | "Search components by task." | query, top_k, type? | ~45 |
| get_component_detail | "Get full component info." | component_id | ~25 |
| install_components | "Install components to .claude/." | component_ids, target_dir? | ~40 |
| check_dependencies | "Check deps and conflicts." | component_ids | ~30 |
| ingest_repo | "Index a component repository." | repo_url | ~25 |

**Total estimated: ~165 tokens** (within 300-token budget with margin)

## Open Questions

Things that couldn't be fully resolved:

1. **Settings.json merge strategy**
   - What we know: Settings are JSON, need deep merge
   - What's unclear: Conflict resolution for same key with different values
   - Recommendation: Last-write-wins with warning log; user can manually resolve

2. **HTTP vs STDIO transport**
   - What we know: Claude Code supports both; STDIO is default for local servers
   - What's unclear: Performance implications for this use case
   - Recommendation: Start with STDIO (simpler setup), add HTTP if needed

3. **Repo ingestion authentication**
   - What we know: ingest_repo tool needs to access potentially private repos
   - What's unclear: How to handle GitHub tokens securely
   - Recommendation: Use git credential helper (user's existing auth), don't store tokens

## Sources

### Primary (HIGH confidence)
- [FastMCP GitHub](https://github.com/jlowin/fastmcp) - Tool definition patterns, server creation
- [MCP Build Server Guide](https://modelcontextprotocol.io/docs/develop/build-server) - Official Python examples
- [Claude Code Skills Docs](https://code.claude.com/docs/en/skills) - .claude/ directory structure

### Secondary (MEDIUM confidence)
- [FastMCP PyPI](https://pypi.org/project/fastmcp/) - Version 2.14.4 confirmed
- [MCP Token Optimization](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1576) - SEP-1576 schema optimization

### Tertiary (LOW confidence)
- WebSearch results on graph-path rationale generation - Academic patterns, not MCP-specific

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastMCP is clearly the standard, verified via official docs
- Architecture: HIGH - Patterns directly from official MCP and FastMCP docs
- Pitfalls: HIGH - STDIO logging issue is documented in official build guide
- Rationale generation: MEDIUM - Pattern derived from existing codebase, not external validation

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (30 days - FastMCP is stable)
