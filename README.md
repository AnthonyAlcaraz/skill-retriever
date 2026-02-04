# Skill Retriever

**Graph-based MCP server for Claude Code component retrieval.**

Given a task description, returns the minimal correct set of components (agents, skills, commands, hooks, MCPs) with all dependencies resolved.

## What Problem Does This Solve?

Claude Code supports custom components stored in `.claude/` directories:
- **Agents** — Specialized AI personas (code reviewer, architect, etc.)
- **Skills** — Reusable prompts and workflows
- **Commands** — Slash commands (`/commit`, `/review`, etc.)
- **Hooks** — Event handlers (pre-commit, post-task, etc.)
- **MCPs** — Model Context Protocol servers
- **Settings** — Configuration files

The problem: **There are thousands of community components scattered across GitHub repos.** Finding the right ones for your task, understanding their dependencies, and ensuring compatibility is painful.

**Skill Retriever solves this by:**
1. Indexing component repositories into a searchable knowledge graph
2. Understanding dependencies between components
3. Returning exactly what you need for a given task (not too much, not too little)
4. Installing them directly into your `.claude/` directory

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                               │
│                                                                  │
│  "I need to add git commit automation"                          │
│                    │                                             │
│                    ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MCP Client (built into Claude Code)          │   │
│  │                                                           │   │
│  │  tools/call: search_components                            │   │
│  │  tools/call: install_components                           │   │
│  │  tools/call: check_dependencies                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ stdio (JSON-RPC)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Skill Retriever MCP Server                    │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Vector    │  │    Graph    │  │      Metadata           │  │
│  │   Store     │  │    Store    │  │       Store             │  │
│  │  (FAISS)    │  │ (NetworkX)  │  │      (JSON)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                │                    │                  │
│         └────────────────┼────────────────────┘                  │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   Retrieval Pipeline  │                          │
│              │                       │                          │
│              │  1. Vector Search     │                          │
│              │  2. Graph PPR         │                          │
│              │  3. Score Fusion      │                          │
│              │  4. Dep Resolution    │                          │
│              │  5. Conflict Check    │                          │
│              │  6. Context Assembly  │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Ingestion (Indexing Repositories)

When you ingest a component repository:

```
Repository (GitHub)
       │
       ▼
┌──────────────────┐
│  Clone to temp   │
└──────────────────┘
       │
       ▼
┌──────────────────┐    Supports:
│  Crawler         │    - davila7/claude-code-cli style (cli-tool/components/)
│  (Strategy-based)│    - .claude/{agents,skills,commands}/ style
└──────────────────┘    - Generic markdown with frontmatter
       │                - Python source files with docstrings
       ▼
┌──────────────────┐
│  Entity Resolver │    Deduplicates similar components using:
│  (Fuzzy + Embed) │    - RapidFuzz token_sort_ratio (Phase 1)
└──────────────────┘    - Embedding cosine similarity (Phase 2)
       │
       ▼
┌──────────────────┐
│  Index into:     │
│  - Graph nodes   │    Component → Node with type, label
│  - Graph edges   │    Dependencies → DEPENDS_ON edges
│  - Vector store  │    Embeddings for semantic search
│  - Metadata      │    Full content for installation
└──────────────────┘
```

### 2. Retrieval (Finding Components)

When you search for components:

```
Query: "git commit automation with conventional commits"
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│                    Query Planning                              │
│                                                                │
│  - Extract entities (keywords, component names)                │
│  - Determine complexity (simple/medium/complex)                │
│  - Decide: use PPR? use flow pruning?                         │
│  - Detect abstraction level (agent vs command vs hook)        │
└───────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐      ┌───────────────────────┐
│ Vector Search │      │ Graph PPR (PageRank)  │
│               │      │                       │
│ Semantic      │      │ Follows dependency    │
│ similarity    │      │ edges to find         │
│ via FAISS     │      │ related components    │
└───────────────┘      └───────────────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
┌───────────────────────────────────────────────────────────────┐
│                    Score Fusion                                │
│                                                                │
│  Combined score = α × vector_score + (1-α) × graph_score      │
│  Filtered by component type if specified                       │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│              Transitive Dependency Resolution                  │
│                                                                │
│  If "commit-command" depends on "git-utils" which depends     │
│  on "shell-helpers" → all three are included                  │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│                  Conflict Detection                            │
│                                                                │
│  Check CONFLICTS_WITH edges between selected components        │
│  Warn if incompatible components would be installed           │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│                  Context Assembly                              │
│                                                                │
│  - Sort by type priority (agents > skills > commands)         │
│  - Estimate token cost per component                          │
│  - Stay within token budget                                   │
│  - Generate rationale for each recommendation                 │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│                      Results                                   │
│                                                                │
│  [                                                            │
│    { id: "davila7/commit-command", score: 0.92,               │
│      rationale: "High semantic match + 3 dependents" },       │
│    { id: "davila7/git-utils", score: 0.85,                    │
│      rationale: "Required dependency of commit-command" }     │
│  ]                                                            │
└───────────────────────────────────────────────────────────────┘
```

### 3. Installation

When you install components:

```
install_components(["davila7/commit-command"])
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│              Resolve Dependencies                              │
│                                                                │
│  commit-command → [git-utils, shell-helpers]                  │
│  Total: 3 components to install                               │
└───────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────┐
│              Write to .claude/                                 │
│                                                                │
│  .claude/                                                     │
│  ├── commands/                                                │
│  │   └── commit.md          ← commit-command                  │
│  └── skills/                                                  │
│      ├── git-utils.md       ← dependency                      │
│      └── shell-helpers.md   ← transitive dependency           │
└───────────────────────────────────────────────────────────────┘
```

### 4. Auto-Sync (SYNC-01, SYNC-02)

Repositories can be tracked for automatic updates:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Sync Manager                                │
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────────────────┐  │
│  │  Webhook Server  │         │       Repo Poller            │  │
│  │  (port 9847)     │         │  (hourly by default)         │  │
│  │                  │         │                              │  │
│  │  POST /webhook   │         │  GET /repos/{owner}/{repo}   │  │
│  │  ← GitHub push   │         │  → GitHub API                │  │
│  └────────┬─────────┘         └──────────────┬───────────────┘  │
│           │                                   │                  │
│           └─────────────┬─────────────────────┘                  │
│                         ▼                                        │
│              ┌─────────────────────┐                            │
│              │  Change Detected?   │                            │
│              │  (new commit SHA)   │                            │
│              └──────────┬──────────┘                            │
│                         │ yes                                    │
│                         ▼                                        │
│              ┌─────────────────────┐                            │
│              │   Re-ingest Repo    │                            │
│              │   (incremental)     │                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## Integration with Claude Code

### Setup

1. **Add to Claude Code's MCP config** (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "skill-retriever": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/skill-retriever", "skill-retriever"]
    }
  }
}
```

2. **Restart Claude Code** to load the MCP server.

### Available Tools

Once configured, Claude Code can use these tools:

| Tool | Purpose |
|------|---------|
| `search_components` | Find components for a task description |
| `get_component_detail` | Get full info about a specific component |
| `install_components` | Install components to `.claude/` |
| `check_dependencies` | Check deps and conflicts before install |
| `ingest_repo` | Index a new component repository |
| `register_repo` | Track a repo for auto-sync |
| `unregister_repo` | Stop tracking a repo |
| `list_tracked_repos` | List all tracked repos |
| `sync_status` | Get sync system status |
| `start_sync_server` | Start webhook + poller |
| `stop_sync_server` | Stop sync services |
| `poll_repos_now` | Trigger immediate poll |

### Example Conversation

```
User: I need to set up git commit automation with conventional commits

Claude: Let me search for relevant components.

[Calls search_components with query="git commit automation conventional commits"]

I found 3 components that would help:

1. **commit-command** (command) - Automated git commits with conventional format
   - Score: 0.92
   - Health: active (updated 2 days ago)
   - Token cost: 450

2. **git-utils** (skill) - Git helper functions
   - Score: 0.85
   - Required by: commit-command

3. **conventional-commits-hook** (hook) - Pre-commit validation
   - Score: 0.78
   - Health: active

Would you like me to install these?

User: Yes, install them

Claude: [Calls install_components with ids=["davila7/commit-command", "davila7/conventional-commits-hook"]]

Installed 4 components to .claude/:
- commands/commit.md
- skills/git-utils.md
- skills/shell-helpers.md (dependency)
- hooks/conventional-commits.md

You can now use `/commit` to create conventional commits!
```

## Data Flow Summary

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   GitHub    │────▶│   Ingestion  │────▶│   Graph Store   │
│   Repos     │     │   Pipeline   │     │   (NetworkX)    │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Claude    │◀───▶│     MCP      │◀───▶│   Retrieval     │
│    Code     │     │   Server     │     │   Pipeline      │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │                      │
                           ▼                      ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │   .claude/   │     │  Vector Store   │
                    │  directory   │     │    (FAISS)      │
                    └──────────────┘     └─────────────────┘
```

## Key Design Decisions

1. **Hybrid retrieval** (vector + graph) — Semantic similarity alone misses dependency relationships
2. **Incremental ingestion** — Only re-index changed files, not entire repos
3. **Entity resolution** — Deduplicate similar components across repos
4. **Token budgeting** — Don't overwhelm Claude's context window
5. **Health signals** — Surface stale/abandoned components
6. **MCP protocol** — Native integration with Claude Code (no plugins needed)

## Requirements Coverage

### v1 (Complete)
- Ingestion: crawl any repo structure, extract metadata + git signals
- Retrieval: semantic search + graph traversal + score fusion
- Dependencies: transitive resolution + conflict detection
- Integration: MCP server + component installation

### v2 (Implemented)
- SYNC-01: Webhook server for GitHub push events
- SYNC-02: Auto-reingest on detected changes
- SYNC-03: Incremental ingestion
- RETR-06: Abstraction level awareness
- LRNG-03: Co-occurrence tracking
- HLTH-01: Component health status

### Deferred
- RETR-05: LLM-assisted query rewriting
- LRNG-01/02: Collaborative filtering from usage patterns
- HLTH-02: Deprecation warnings

## Development

```bash
# Install
uv sync

# Run MCP server
uv run skill-retriever

# Run tests
uv run pytest

# Type check
uv run pyright

# Lint
uv run ruff check
```

## License

MIT
