---
name: ui-tars-desktop
description: ByteDance multimodal AI agent stack with GUI Agent, browser automation, and MCP integration
version: "1.0"
author: ByteDance
tags:
  - multimodal
  - gui-agent
  - browser-automation
  - desktop-agent
  - mcp
  - computer-use
tools:
  - agent-tars-cli
  - ui-tars-desktop
  - ui-tars-sdk
---

# TARS: Multimodal AI Agent Stack

## Overview
TARS is ByteDance's open-source multimodal AI agent stack shipping two products: **Agent TARS** (a general multimodal AI agent with CLI and Web UI) and **UI-TARS Desktop** (a native GUI agent application based on the UI-TARS vision model). Agent TARS brings GUI Agent capabilities, vision understanding, and MCP tool integration into your terminal, computer, and browser. It supports task completion workflows closer to human behavior through cutting-edge multimodal LLMs.

## Installation

### Agent TARS CLI
```bash
npm install -g @agent-tars/cli
# Requires Node.js 18+

# Run the CLI
agent-tars --help
```

### UI-TARS Desktop
Download from [GitHub Releases](https://github.com/bytedance/UI-TARS-desktop/releases) or build from source:
```bash
git clone https://github.com/bytedance/UI-TARS-desktop.git
cd UI-TARS-desktop
pnpm install
pnpm build
```

### UI-TARS SDK (for custom agents)
```bash
npm install @anthropic-ai/sdk  # or use UI-TARS model API
```

## Key Commands / Usage

### Agent TARS CLI
```bash
# Start with default configuration
agent-tars

# Run with specific model
agent-tars --model claude-sonnet-4-5-20250929

# Run with MCP servers
agent-tars --config ./tars-config.json
```

### Task Examples
```
"Book the earliest flight from San Jose to New York on September 1st on Priceline"
"Generate a sales chart from the CSV on my desktop"
"Fill out the job application form on LinkedIn"
```

### UI-TARS Desktop Operators
- **Local Operator**: Controls your local computer directly
- **Remote Computer Operator**: Controls remote machines
- **Remote Browser Operator**: Automates browsers on remote servers

## Architecture
Agent TARS has a modular architecture: (1) **Multimodal Perception** - screenshots + vision model for screen understanding, (2) **Action Planning** - LLM-based reasoning with chain-of-thought for complex tasks, (3) **Tool Execution** - MCP server integration for shell, browser, filesystem, and custom tools, (4) **AIO Sandbox** - isolated execution environment for safe tool runs. The CLI supports streaming output, timing statistics per tool call, and an Event Stream Viewer for debugging. UI-TARS Desktop wraps the UI-TARS vision model (trained specifically for GUI understanding) in an Electron app with local and remote operators.

## OS Agent Integration
TARS provides a production-ready agent stack for both developers (CLI) and end users (Desktop app). Key differentiators: (1) native MCP integration means any MCP server extends the agent's capabilities, (2) the AIO Sandbox provides isolated execution for safety, (3) remote operators enable controlling machines over the network, (4) the UI-TARS vision model is purpose-trained for GUI understanding, outperforming general-purpose models on UI tasks. Combines well with other OS agents as a vision+planning frontend.

## Source
- Repository: https://github.com/bytedance/UI-TARS-desktop
- Stars: 27,785
- Language: TypeScript
- Documentation: https://agent-tars.com
