# Skill Retriever

**Graph-based MCP server for Claude Code component retrieval.**

Given a task description, returns the minimal correct set of components (agents, skills, commands, hooks, MCPs) with all dependencies resolved.

## Current Index

**1,027 components** from 23 repositories, auto-discovered and synced.

| Type | Count | Description |
|------|-------|-------------|
| **Skills** | 529 | Portable instruction sets that package domain expertise and procedural knowledge |
| **Agents** | 419 | Specialized AI personas with isolated context and fine-grained permissions |
| **Commands** | 36 | Slash commands (`/commit`, `/review`, etc.) |
| **Hooks** | 20 | Event handlers (SessionStart, PreCompact, etc.) |
| **MCPs** | 20 | Model Context Protocol servers for external integrations |
| **Settings** | 3 | Configuration presets |

### Top Repositories

| Repository | Components | Description |
|------------|------------|-------------|
| [wshobson/agents](https://github.com/wshobson/agents) | 221 | Multi-agent orchestration with 129 skills |
| [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) | 172 | 200+ curated skills compatible with Codex, Gemini CLI |
| [davepoon/buildwithclaude](https://github.com/davepoon/buildwithclaude) | 152 | Full-stack development skills |
| [BehiSecc/awesome-claude-skills](https://github.com/BehiSecc/awesome-claude-skills) | 61 | Document processing, security, scientific skills |
| [anthropics/skills](https://github.com/anthropics/skills) | 17 | Official Anthropic skills (Excel, PowerPoint, PDF, skill-creator) |
| [obra/superpowers](https://github.com/obra/superpowers) | 13 | TDD, debugging, and software development methodology |

## What Problem Does This Solve?

Claude Code supports custom components stored in `.claude/` directories.

### The Agent Skills Standard

Skills are **folders of instructions** that extend Claude's capabilities. Every skill includes a `SKILL.md` markdown file containing name, description, and instructions. Skills are **progressively disclosed**—only name and description load initially; full instructions load only when triggered.

The open standard means skills work across:
- Claude AI and Claude Desktop
- Claude Code
- Claude Agent SDK
- Codex, Gemini CLI, OpenCode, and other compatible platforms

### Component Types Explained

| Type | What It Does | When to Use |
|------|-------------|-------------|
| **Skill** | Packages domain expertise + procedural knowledge into portable instructions | Repeatable workflows, company-specific analysis, new capabilities |
| **Agent** | Spawned subprocess with isolated context and tool access | Parallel execution, specialized tasks, permission isolation |
| **Command** | Slash command (`/name`) that triggers specific behavior | Quick actions, shortcuts, task invocation |
| **Hook** | Runs automatically on events (SessionStart, PreCompact) | Context setup, auto-save, cleanup |
| **MCP** | Model Context Protocol server connecting to external systems | Database access, APIs, file systems |

### Skills vs Tools vs Subagents

| Concept | Analogy | Persistence | Context |
|---------|---------|-------------|---------|
| **Tools** | Hammer, saw, nails | Always in context | Adds to main window |
| **Skills** | How to build a bookshelf | Progressively loaded | Name/desc → SKILL.md → refs |
| **Subagents** | Hire a specialist | Session-scoped | Isolated from parent |

**Key insight**: Skills solve the context window problem. By progressively disclosing instructions, they avoid polluting context with data that may never be needed.

### The Problem This Solves

**There are now 800+ community components scattered across GitHub repos.** Finding the right ones for your task, understanding their dependencies, and ensuring compatibility is painful.

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
┌──────────────────┐    Strategies (first match wins):
│  Crawler         │    1. Davila7Strategy: cli-tool/components/{type}/
│  (Strategy-based)│    2. PluginMarketplaceStrategy: plugins/{name}/skills/
└──────────────────┘    3. FlatDirectoryStrategy: .claude/{type}/
       │                4. GenericMarkdownStrategy: Any *.md with name frontmatter
       │                5. AwesomeListStrategy: README.md curated lists
       │                6. PythonModuleStrategy: *.py with docstrings
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

### 4. Discovery Pipeline (OSS-01, HEAL-01)

Automatically discovers and ingests high-quality skill repositories from GitHub:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Discovery Pipeline                            │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │   OSS Scout      │  Searches GitHub for skill repos:         │
│  │                  │  - 8 search queries (claude, skills, etc) │
│  │  discover()      │  - MIN_STARS: 5                           │
│  │  ─────────────▶  │  - Recent activity: 180 days              │
│  └────────┬─────────┘  - Quality scoring (stars, topics, etc)   │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Filter & Score  │  Score = stars (40) + recency (20)        │
│  │                  │        + topics (20) + description (10)   │
│  │  min_score: 30   │        + forks (10)                       │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │    Ingest New    │  Clone → Crawl → Dedupe → Index           │
│  │  (max 10/run)    │  Uses same pipeline as ingest_repo        │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │   Auto-Healer    │  Tracks failures:                         │
│  │                  │  - CLONE_FAILED, NO_COMPONENTS            │
│  │  MAX_RETRIES: 3  │  - NETWORK_ERROR, RATE_LIMITED            │
│  └──────────────────┘  Automatically retries healable failures   │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Auto-Sync (SYNC-01, SYNC-02)

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

### 6. Feedback Loop (LRNG-04, LRNG-05, LRNG-06)

Execution outcomes feed back into the graph to improve future recommendations:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Feedback Loop                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Outcome Tracking (LRNG-05)               │   │
│  │                                                          │   │
│  │  install_components()                                    │   │
│  │         │                                                │   │
│  │         ├── success → INSTALL_SUCCESS + bump usage       │   │
│  │         └── failure → INSTALL_FAILURE + track context    │   │
│  │                                                          │   │
│  │  report_outcome()                                        │   │
│  │         ├── USED_IN_SESSION → usage count++              │   │
│  │         ├── REMOVED_BY_USER → negative feedback          │   │
│  │         └── DEPRECATED → deprecation flag                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Usage-Based Boosting (LRNG-04)              │   │
│  │                                                          │   │
│  │  Selection Rate Boost:                                   │   │
│  │    high_selection_rate → +50% score boost                │   │
│  │    low_selection_rate  → no boost                        │   │
│  │                                                          │   │
│  │  Co-Selection Boost:                                     │   │
│  │    frequently_selected_together → +10% each (max 30%)    │   │
│  │                                                          │   │
│  │  Final score = base_score × boost_factor                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Feedback Engine (LRNG-06)                   │   │
│  │                                                          │   │
│  │  analyze_feedback() discovers patterns:                  │   │
│  │                                                          │   │
│  │  Co-selections (≥3) → suggest BUNDLES_WITH edge          │   │
│  │  Co-failures (≥2)   → suggest CONFLICTS_WITH edge        │   │
│  │                                                          │   │
│  │  Human reviews suggestions via review_suggestion()       │   │
│  │  Accepted suggestions → apply_feedback_suggestions()     │   │
│  │  New edges added to graph with confidence scores         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: The system learns from real-world usage. Components that work well together get boosted. Components that fail together get flagged as conflicts. This creates a self-improving recommendation engine.

### 7. Security Scanning (SEC-01)

Scans components for security vulnerabilities during ingestion and on-demand:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Scanner                              │
│                                                                  │
│  Based on Yi Liu et al. "Agent Skills in the Wild" research:    │
│  - 26.1% of skills contain vulnerable patterns                  │
│  - 5.2% show malicious intent indicators                        │
│  - Skills with scripts are 2.12x more likely to be vulnerable   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Vulnerability Detection                      │   │
│  │                                                          │   │
│  │  Data Exfiltration (13.3%)                               │   │
│  │    - HTTP POST with data payload                         │   │
│  │    - File read + external request                        │   │
│  │    - Webhook endpoints                                   │   │
│  │                                                          │   │
│  │  Credential Access                                       │   │
│  │    - Environment variable harvesting                     │   │
│  │    - SSH key / AWS credential access                     │   │
│  │    - Sensitive env vars (API_KEY, SECRET, TOKEN)        │   │
│  │                                                          │   │
│  │  Privilege Escalation (11.8%)                            │   │
│  │    - Shell injection via variable interpolation          │   │
│  │    - Dynamic code execution (eval/exec)                  │   │
│  │    - sudo execution, chmod 777                           │   │
│  │    - Download and execute patterns                       │   │
│  │                                                          │   │
│  │  Obfuscation (malicious intent)                          │   │
│  │    - Hex-encoded strings                                 │   │
│  │    - Unicode escapes                                     │   │
│  │    - String concatenation obfuscation                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Risk Assessment                              │   │
│  │                                                          │   │
│  │  Risk Levels: safe → low → medium → high → critical      │   │
│  │                                                          │   │
│  │  Risk Score (0-100):                                     │   │
│  │    Base = sum of finding weights                         │   │
│  │    Script multiplier = 1.5x if has_scripts               │   │
│  │                                                          │   │
│  │  Each component stores:                                  │   │
│  │    - security_risk_level                                 │   │
│  │    - security_risk_score                                 │   │
│  │    - security_findings_count                             │   │
│  │    - has_scripts                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Integration Points                           │   │
│  │                                                          │   │
│  │  Ingestion: scan during ingest_repo()                    │   │
│  │  Retrieval: include SecurityStatus in search results     │   │
│  │  On-demand: security_scan() and security_audit() tools   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Security scanning catches 22%+ of potentially vulnerable patterns before they reach your codebase. The system flags data exfiltration, credential access, privilege escalation, and code obfuscation.

**Current Index Statistics:**
| Risk Level | Count | % |
|------------|-------|---|
| Safe | 796 | 77.5% |
| Low | 2 | 0.2% |
| Medium | 19 | 1.9% |
| High | 8 | 0.8% |
| Critical | 202 | 19.7% |

**Top Finding Patterns (in CRITICAL components):**
| Pattern | Count | Notes |
|---------|-------|-------|
| `shell_injection` | 424 | Many are bash examples in markdown (false positives) |
| `webhook_post` | 87 | Discord/Slack webhook URLs |
| `env_harvest_all` | 74 | `process.env` / `os.environ` access |
| `ssh_key_access` | 51 | References to `.ssh/` paths |
| `http_post_with_data` | 38 | HTTP POST with data payload |

**Known Limitations:**
- The `shell_injection` pattern has false positives for bash code blocks in markdown
- Webhook patterns flag legitimate integrations (Discord bots, Slack notifications)
- Future: LLM-assisted false positive reduction (SEC-02)

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
| **Search & Install** | |
| `search_components` | Find components for a task description |
| `get_component_detail` | Get full info about a specific component |
| `install_components` | Install components to `.claude/` (auto-records outcome) |
| `check_dependencies` | Check deps and conflicts before install |
| **Ingestion** | |
| `ingest_repo` | Index a new component repository |
| **Sync Management** | |
| `register_repo` | Track a repo for auto-sync |
| `unregister_repo` | Stop tracking a repo |
| `list_tracked_repos` | List all tracked repos |
| `sync_status` | Get sync system status |
| `start_sync_server` | Start webhook + poller |
| `stop_sync_server` | Stop sync services |
| `poll_repos_now` | Trigger immediate poll |
| **Discovery Pipeline** | |
| `run_discovery_pipeline` | Discover + ingest new skill repos from GitHub |
| `discover_repos` | Search GitHub for skill repositories |
| `get_pipeline_status` | Get discovery pipeline configuration |
| `get_heal_status` | View auto-heal failures and status |
| `clear_heal_failures` | Clear tracked failures |
| **Outcome Tracking** | |
| `report_outcome` | Record usage outcome (used, removed, deprecated) |
| `get_outcome_stats` | Get success/failure stats for a component |
| `get_outcome_report` | View problematic components and conflicts |
| **Feedback Engine** | |
| `analyze_feedback` | Analyze patterns to suggest graph improvements |
| `get_feedback_suggestions` | View pending edge suggestions |
| `review_suggestion` | Accept or reject a suggested edge |
| `apply_feedback_suggestions` | Apply accepted suggestions to the graph |
| **Security Scanning** | |
| `security_scan` | Scan a specific component for vulnerabilities |
| `security_audit` | Audit all components, report by risk level |
| `backfill_security_scans` | Scan existing components that haven't been scanned |

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

### Workflow with Security Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                 Claude Code + Skill Retriever Workflow           │
│                                                                  │
│  1. USER: "I need JWT authentication"                           │
│                    │                                             │
│                    ▼                                             │
│  2. CLAUDE: search_components("JWT authentication")              │
│                    │                                             │
│                    ▼                                             │
│  3. SKILL RETRIEVER returns:                                     │
│     ┌────────────────────────────────────────────────────┐      │
│     │ auth-jwt-skill                                      │      │
│     │   Score: 0.89                                       │      │
│     │   Health: active (2 days ago)                       │      │
│     │   Security: ⚠️ MEDIUM (env_sensitive_keys)          │      │
│     │   Tokens: 320                                       │      │
│     │                                                     │      │
│     │ crypto-utils                                        │      │
│     │   Score: 0.72                                       │      │
│     │   Health: active                                    │      │
│     │   Security: ✅ SAFE                                 │      │
│     │   Tokens: 180                                       │      │
│     └────────────────────────────────────────────────────┘      │
│                    │                                             │
│                    ▼                                             │
│  4. CLAUDE: "auth-jwt-skill has MEDIUM security risk             │
│              (accesses JWT_SECRET from env). Proceed?"           │
│                    │                                             │
│                    ▼                                             │
│  5. USER: "Yes, that's expected for JWT"                        │
│                    │                                             │
│                    ▼                                             │
│  6. CLAUDE: install_components(["auth-jwt-skill"])               │
│                    │                                             │
│                    ▼                                             │
│  7. SKILL RETRIEVER:                                             │
│     - Resolves dependencies (adds crypto-utils)                  │
│     - Writes to .claude/skills/                                  │
│     - Records INSTALL_SUCCESS outcome                            │
│                    │                                             │
│                    ▼                                             │
│  8. CLAUDE: "Installed auth-jwt-skill + crypto-utils.            │
│              Note: Requires JWT_SECRET env variable."            │
└─────────────────────────────────────────────────────────────────┘
```

### Security-Aware Retrieval

When `search_components` returns results, each component includes:

```json
{
  "id": "owner/repo/skill/auth-jwt",
  "name": "auth-jwt",
  "type": "skill",
  "score": 0.89,
  "rationale": "High semantic match + required dependency",
  "token_cost": 320,
  "health": {
    "status": "active",
    "last_updated": "2026-02-02T10:30:00Z",
    "commit_frequency": "high"
  },
  "security": {
    "risk_level": "medium",
    "risk_score": 25.0,
    "findings_count": 1,
    "has_scripts": false
  }
}
```

**Best Practice**: Claude should surface security warnings to users before installation, especially for CRITICAL and HIGH risk components.

### Backfilling Existing Components

If you have components indexed before SEC-01 was implemented:

```
User: Run a security audit on all components

Claude: [Calls security_audit(risk_level="medium")]

Security Audit Results:
- Total: 1027 components
- Safe: 796 (77.5%)
- Low: 2 (0.2%)
- Medium: 19 (1.9%)
- High: 8 (0.8%)
- Critical: 202 (19.7%)

Would you like to see the flagged components?

User: Yes, show critical ones

Claude: [Shows list of critical components with their findings]

Note: Many "shell_injection" findings are false positives from
bash code examples in markdown. Review manually for true concerns.
```

To backfill security scans for components indexed before SEC-01:
```
Claude: [Calls backfill_security_scans(force_rescan=false)]
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
7. **Security-first scanning** — 26% of skills contain vulnerabilities; scan before installation

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
- OSS-01: GitHub-based repository discovery (OSS Scout)
- HEAL-01: Auto-heal for failed ingestions with retry logic
- RETR-06: Abstraction level awareness
- RETR-07: Fuzzy entity extraction with RapidFuzz + synonym expansion
- LRNG-03: Co-occurrence tracking
- LRNG-04: Usage-based score boosting (selection rate + co-selection)
- LRNG-05: Outcome tracking (install success/failure, usage, removal)
- LRNG-06: Feedback engine for implicit edge discovery
- HLTH-01: Component health status
- SEC-01: Security vulnerability scanning (based on Yi Liu et al. research)

### Deferred
- RETR-05: LLM-assisted query rewriting
- LRNG-01/02: Collaborative filtering from usage patterns
- HLTH-02: Deprecation warnings
- SEC-02: LLM-assisted false positive reduction for security scanning
- SEC-03: Real-time re-scanning of installed components

## Troubleshooting

### Ingestion Failures

```bash
# Check auto-heal status
get_heal_status()
```

| Failure Type | Cause | Solution |
|-------------|-------|----------|
| `CLONE_FAILED` | Network timeout, auth required | Check URL, verify public access |
| `NO_COMPONENTS` | Repo has no Claude Code components | Expected for non-skill repos |
| `RATE_LIMITED` | GitHub API limit exceeded | Wait 1 hour, retry |
| `PARSE_ERROR` | Malformed markdown/YAML | Open issue on source repo |

**To retry failed ingestion:**
```bash
clear_heal_failures()
ingest_repo(repo_url="https://github.com/owner/repo", incremental=False)
```

### Search Returns Empty Results

1. **Verify index is loaded:**
   ```bash
   sync_status()  # Check tracked_repos > 0
   ```

2. **Check if component exists:**
   ```bash
   get_component_detail(component_id="owner/repo/skill/name")
   ```

3. **Try broader search terms:**
   - "auth" instead of "JWT RS256 authentication"
   - Remove specific technology mentions

4. **Check type filter isn't too restrictive:**
   ```bash
   search_components(query="auth", component_type=None)  # Remove filter
   ```

### Installation Failures

```bash
# Always check dependencies first
check_dependencies(component_ids=["id1", "id2"])
```

| Error | Cause | Solution |
|-------|-------|----------|
| Component not found | Not in metadata store | `ingest_repo()` the source repo |
| Conflict detected | Incompatible components | Choose one, or use `conflicts` field to understand |
| Write permission denied | Target dir not writable | Check `.claude/` exists and is writable |

### Security Scan False Positives

The `shell_injection` pattern flags many legitimate bash examples:

```bash
# This is flagged but safe (bash in markdown):
gh pr view $PR_NUMBER

# This would be actually dangerous:
eval "$USER_INPUT"
```

**To review false positives:**
```bash
security_scan(component_id="owner/repo/skill/name")
# Review each finding's matched_text
```

### MCP Server Won't Start

1. **Check Python version:** Requires 3.11+
2. **Check dependencies:** `uv sync`
3. **Check port conflicts:** Webhook server uses 9847
4. **Check Claude Code config:**
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

### Data Corruption

If the index seems corrupted:

```bash
# Backup existing data
cp -r ~/.skill-retriever/data ~/.skill-retriever/data.bak

# Clear and re-ingest
rm ~/.skill-retriever/data/*.json
rm -rf ~/.skill-retriever/data/vectors/

# Re-run discovery pipeline
run_discovery_pipeline(dry_run=False, max_new_repos=50)
```

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

## Related Resources

- **[DeepLearning.AI Agent Skills Course](https://learn.deeplearning.ai/courses/agent-skills-with-anthropic)** — Official course covering skill creation, Claude API, Claude Code, and Agent SDK
- **[anthropics/skills](https://github.com/anthropics/skills)** — Official Anthropic skills repository
- **[Agent Skills Specification](https://agentskills.io)** — Open standard documentation

## License

MIT
