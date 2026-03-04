---
name: multi-mcp-agent
description: Cross-platform MCP agent integrating GitHub, Gmail, Calendar, and Perplexity via Agno framework
version: "1.0"
author: Shubhamsaboo
tags:
  - mcp
  - multi-agent
  - github
  - gmail
  - calendar
  - perplexity
  - productivity
tools:
  - agno
  - mcp-github
  - mcp-gmail
  - mcp-calendar
  - mcp-perplexity
---

# Multi-MCP Intelligent Assistant

## Overview
The Multi-MCP Intelligent Assistant integrates multiple Model Context Protocol servers to provide seamless access to GitHub, Perplexity, Calendar, and Gmail through natural language. Built on the Agno AI Agent framework with GPT-4o, it enables cross-platform workflow automation: create a GitHub issue and schedule a follow-up meeting, research a topic and email a summary, review pull requests and update the calendar. Features conversation memory, tool chaining, and streaming responses.

## Installation

```bash
# Clone the repository
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/mcp_ai_agents/multi_mcp_agent

# Install dependencies
pip install -r requirements.txt

# Verify Node.js (required for MCP servers)
node --version && npm --version && npx --version

# Configure environment
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key
GITHUB_PERSONAL_ACCESS_TOKEN=your-github-token
PERPLEXITY_API_KEY=your-perplexity-api-key
EOF
```

**Requirements:** Python 3.10+, Node.js + npm, API keys for OpenAI, GitHub, Perplexity.

## Key Commands / Usage

### Run the Agent
```bash
python multi_mcp_agent.py
```

### GitHub Operations
```
"Show my recent GitHub repositories"
"Create a new issue titled 'Fix login bug' in my-project repo"
"Search for Python code across my repositories"
"Review the latest pull requests on my-app"
```

### Research
```
"Search for the latest developments in AI agents"
"Find documentation for the Agno framework"
"What are the trending topics in machine learning this week?"
```

### Calendar & Email
```
"Schedule a meeting for next Tuesday at 2pm"
"Show my upcoming appointments for this week"
"Send a summary email of today's research findings"
```

### Cross-Platform Workflows
```
"Create a GitHub issue for the bug we discussed, then schedule a fix meeting"
"Research transformer architectures and create a summary document"
"Find trending repositories and add review meetings to my calendar"
```

## Architecture
The system uses the Agno framework for agent orchestration with four MCP server integrations: (1) **GitHub MCP** - repository management, issues, PRs, code search via `@modelcontextprotocol/server-github`, (2) **Perplexity MCP** - real-time web search and research, (3) **Calendar MCP** - event scheduling and availability management, (4) **Gmail MCP** - email reading, composing, and organization. The agent uses GPT-4o for reasoning, maintains session memory for context retention, and chains tools for complex workflows. Each MCP server runs as a separate Node.js process.

## OS Agent Integration
Multi-MCP Agent demonstrates how MCP protocol enables composable AI assistants. Instead of building monolithic agents, MCP allows plugging together specialized servers for different capabilities. Integration patterns: (1) add this agent's MCP servers to any MCP-compatible client (Claude Code, Cursor), (2) extend with custom MCP servers for domain-specific tools, (3) combine with desktop/browser agents for full computer + cloud service automation. The Agno framework handles agent orchestration, memory, and tool routing.

## Source
- Repository: https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/mcp_ai_agents/multi_mcp_agent
- Parent Repo Stars: 93,998
- Language: Python
