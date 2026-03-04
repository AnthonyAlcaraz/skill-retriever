---
name: oracle
description: Multi-model AI consultation -- bundle prompts and files, query GPT-5.1 Pro, Gemini 3, Claude 4, and more
version: "1.0"
author: steipete
tags:
  - multi-model
  - ai-consultation
  - gpt-5
  - gemini
  - claude
  - code-review
  - mcp
tools:
  - oracle (CLI)
  - oracle-mcp (MCP server)
  - oracle tui
  - oracle status
  - oracle session
  - oracle restart
  - oracle serve
---

# Oracle - Multi-Model AI Consultation

## Overview

Oracle bundles your prompt and files so another AI model can answer with real context. It speaks GPT-5.1 Pro, GPT-5.2, Gemini 3 Pro, Claude Sonnet 4.5, Claude Opus 4.1, and more -- and can query multiple models in a single run with aggregated cost and usage reporting. When API keys are unavailable, Oracle falls back to browser automation that opens ChatGPT or Gemini in Chrome using your existing cookies.

The primary use case: when you are stuck, need a second opinion, or want cross-model validation. Oracle packages the relevant code files, applies smart glob/exclude filters, and sends the bundle to one or more models. It supports API mode, browser mode, render/copy mode for manual paste, and an MCP server for integration with other agents.

## Installation

```bash
# npm (global)
npm install -g @steipete/oracle

# Homebrew
brew install steipete/tap/oracle

# Or use directly
npx -y @steipete/oracle --help
```

Requires Node 22+. API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.

## Key Commands / Usage

```bash
# Basic API run (uses OPENAI_API_KEY)
oracle -p "Review the TS data layer for schema drift" --file "src/**/*.ts"

# Multi-model run
oracle -p "Cross-check the data layer" --models gpt-5.1-pro,gemini-3-pro --file "src/**/*.ts"

# Copy bundle for manual paste into ChatGPT
oracle --render --copy -p "Review this code" --file "src/**/*.ts"

# Browser mode (no API key, opens ChatGPT)
oracle --engine browser -p "Walk through the UI smoke test" --file "src/**/*.ts"

# Gemini browser mode (uses Chrome cookies)
oracle --engine browser --model gemini-3-pro --prompt "analyze this" --file src/

# Session management
oracle status --hours 72
oracle session <id> --render
oracle restart <id>

# MCP server mode
oracle-mcp

# Remote browser service
oracle serve  # on a signed-in host
oracle --remote-host host:port --remote-token xyz -p "review" --file src/
```

## Architecture

Oracle's bundling system collects files via glob patterns, applies size guards and exclusion rules, and packages everything into a structured prompt. API mode sends this directly to the model's API with cost tracking. Browser mode launches or attaches to a Chrome instance, navigates to ChatGPT/Gemini, pastes the bundle, and captures the response (with auto-reattach for long GPT-5 Pro runs). The MCP server (`oracle-mcp`) exposes the same functionality over stdio for integration with MCPorter and other MCP clients.

Sessions are persisted in `~/.oracle/sessions` with full replay capability. GPT-5 Pro API runs detach by default and can be reattached via `oracle session <id>`.

## Integration with OS Agent

Oracle serves as the "second opinion" tool in an OS agent stack. When an agent encounters ambiguity, complex bugs, or needs architectural review, it can route the question through Oracle to get perspectives from multiple frontier models. The MCP server mode makes this seamless: configure Oracle in MCPorter, and any agent can call it as a standard MCP tool. The multi-model capability is particularly valuable for reducing hallucination risk through cross-model consensus.

## Source

- Repository: https://github.com/steipete/oracle
- Stars: 1406
- Language: TypeScript
