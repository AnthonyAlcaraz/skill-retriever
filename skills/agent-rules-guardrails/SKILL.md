---
name: agent-rules-guardrails
description: Reusable agent instruction patterns, guardrails, and helper scripts for AI coding agents
version: "1.0"
author: steipete (Peter Steinberger)
tags:
  - guardrails
  - agent-instructions
  - best-practices
  - committer
  - docs-lister
  - browser-tools
  - agent-scripts
tools:
  - committer
  - docs-list
  - browser-tools
---

# Agent Rules & Guardrails

## Overview
Agent-scripts is the canonical mirror of Peter Steinberger's guardrail helpers for AI coding agents. It collects reusable instruction patterns (AGENTS.MD files), helper scripts (committer, docs-list), and browser automation tools shared across multiple repositories. The approach uses a pointer-style AGENTS.MD where downstream repos reference the shared rules instead of duplicating them. Also includes curated skills from Dimillian's collection (Swift concurrency, SwiftUI performance, etc.).

## Installation

```bash
# Clone the repo
git clone https://github.com/steipete/agent-scripts.git
cd agent-scripts

# The scripts are dependency-free and portable
# No build step required for the shell scripts

# Build browser-tools binary (optional, requires Bun)
bun build scripts/browser-tools.ts --compile --outfile bin/browser-tools

# Build docs-list binary (optional, requires Bun)
bun build scripts/docs-list.ts --compile --outfile bin/docs-list
```

## Key Commands / Usage

### Pointer-Style AGENTS.MD
Add this line to any repo's AGENTS.MD:
```
READ ~/Projects/agent-scripts/AGENTS.MD BEFORE ANYTHING (skip if missing).
```
Then append repo-specific rules below. No more duplicating shared guardrail text.

### Committer Helper
```bash
# Stage specific files and commit with enforced non-empty message
./scripts/committer file1.ts file2.ts "Fix authentication flow"
```

### Docs Lister
```bash
# Walk docs/ directory, enforce front-matter (summary, read_when), print summaries
npx tsx scripts/docs-list.ts
# Or use compiled binary
./bin/docs-list
```

### Browser Tools
```bash
# Launch Chrome with DevTools
./bin/browser-tools start --profile default

# Navigate
./bin/browser-tools nav "https://example.com"

# Execute JavaScript
./bin/browser-tools eval 'document.title'

# Take screenshot
./bin/browser-tools screenshot

# Search page content
./bin/browser-tools search --content "login button"

# Get page content as markdown
./bin/browser-tools content "https://example.com"

# Clean up
./bin/browser-tools kill --all --force
```

### Included Skills (from Dimillian)
- `skills/swift-concurrency-expert` - Swift async/await patterns
- `skills/swiftui-liquid-glass` - SwiftUI glass effects
- `skills/swiftui-performance-audit` - SwiftUI performance profiling
- `skills/swiftui-view-refactor` - SwiftUI view decomposition

## Architecture
The system follows a hub-and-spoke model: (1) **Hub** (this repo) contains the canonical AGENTS.MD with shared rules and tool definitions, (2) **Spokes** (downstream repos) contain a one-line pointer to the hub, (3) **Helper Scripts** are dependency-free TypeScript/Bash that run in isolation across repos, (4) **Sync Protocol** ensures changes propagate bidirectionally (edit in any repo, copy back to hub, push to all spokes). Browser-tools is a standalone Chrome DevTools helper inspired by Mario Zechner's "What if you don't need MCP?" approach.

## OS Agent Integration
Agent-scripts provides the instruction and guardrail layer for OS-level AI agents. Integration patterns: (1) use the AGENTS.MD template as a starting point for any new agent project's system instructions, (2) adopt the committer helper for safe, auditable git operations by agents, (3) use browser-tools as a lightweight alternative to full MCP browser servers when you just need Chrome DevTools access, (4) the pointer-style architecture scales across many repos without instruction drift, (5) the docs-list approach ensures agents discover and read relevant documentation before making changes.

## Source
- Repository: https://github.com/steipete/agent-scripts
- Stars: 1,786
- Language: TypeScript
- Also see: https://github.com/Dimillian/Skills
