---
name: agent-scripts
description: Shared guardrail scripts, browser tools, and agent automation helpers for safe terminal and agent operations
version: "1.0"
author: steipete
tags:
  - agent-guardrails
  - terminal-automation
  - browser-tools
  - committer
  - docs-lister
  - safe-operations
tools:
  - scripts/committer
  - scripts/docs-list.ts
  - bin/browser-tools
  - bin/docs-list
---

# Agent Scripts - Safe Terminal and Agent Automation

## Overview

Agent Scripts is steipete's canonical repository of shared guardrail helpers, browser tools, and automation scripts used across all his agent-powered projects. It provides a committer helper that enforces safe git operations, a docs lister that validates documentation front-matter, browser tools for Chrome DevTools automation without the full Oracle CLI, and the pointer-style AGENTS.MD system that keeps agent instructions synchronized across multiple repositories.

This is the foundation layer that makes agent operations safe and reproducible. Rather than giving agents unrestricted terminal access, these scripts provide structured, validated operations with guard rails.

## Installation

```bash
git clone https://github.com/steipete/agent-scripts.git
# Or reference as a pointer from other repos:
# Add to AGENTS.MD: "READ ~/Projects/agent-scripts/AGENTS.MD BEFORE ANYTHING (skip if missing)."
```

Browser tools binary:
```bash
cd agent-scripts
bun build scripts/browser-tools.ts --compile --target bun --outfile bin/browser-tools
```

## Key Commands / Usage

**Committer Helper** - Safe git commits with validation:
```bash
# Stages exactly the files you list, enforces non-empty messages
scripts/committer "file1.ts file2.ts" "feat: add authentication flow"
```

**Docs Lister** - Validates documentation front-matter:
```bash
# Walks docs/, enforces summary + read_when front-matter
bin/docs-list
# Or via pnpm
pnpm run docs:list
```

**Browser Tools** - Chrome DevTools automation:
```bash
# Start Chrome with debugging
bin/browser-tools start --profile default

# Navigate and capture
bin/browser-tools nav https://example.com
bin/browser-tools screenshot
bin/browser-tools eval 'document.title'

# Search page content
bin/browser-tools search --content "login button"

# Inspect DOM
bin/browser-tools inspect

# Clean up
bin/browser-tools kill --all --force
```

**AGENTS.MD Sync** - Keep agent instructions consistent:
```bash
# Canonical instructions live in agent-scripts/AGENTS.MD
# Downstream repos use pointer: "READ ~/Projects/agent-scripts/AGENTS.MD BEFORE ANYTHING"
```

## Architecture

The repository is designed to be dependency-free and portable. Every script runs in isolation across repos without tsconfig path aliases or shared imports. The committer helper wraps git operations with validation. The docs lister uses tsx or a compiled Bun binary. Browser tools launch or inspect DevTools-enabled Chrome profiles using the remote debugging protocol, supporting start, navigate, eval, screenshot, search, content extraction, inspect, and kill commands.

The pointer-style AGENTS.MD system eliminates instruction drift: the canonical copy lives here, downstream repos contain only a pointer line plus repo-specific additions.

## Integration with OS Agent

Agent Scripts provides the safety layer for terminal operations in an OS agent. Instead of raw shell access, agents use the committer for git operations (preventing empty commits, wrong files), the docs lister for documentation validation, and browser tools for lightweight Chrome automation. The AGENTS.MD pointer system ensures every agent session starts with the correct guardrails regardless of which repository it operates in. This complements MCPorter (MCP orchestration), Peekaboo (GUI automation), and Claude Code MCP (code editing) by providing the operational safety net.

## Source

- Repository: https://github.com/steipete/agent-scripts
- Stars: 1786
- Language: TypeScript
