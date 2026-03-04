---
name: browser-mcp-agent
description: Natural language browser control via MCP protocol with Playwright and Streamlit UI
version: "1.0"
author: Shubhamsaboo
tags:
  - browser-automation
  - mcp
  - playwright
  - natural-language
  - web-scraping
  - streamlit
tools:
  - mcp-agent
  - playwright
  - streamlit
---

# Browser MCP Agent

## Overview
Browser MCP Agent is a Streamlit application that enables natural language browser control through the Model Context Protocol (MCP). It uses MCP-Agent with Playwright integration to translate English commands into browser actions: navigating websites, clicking buttons, filling forms, scrolling content, taking screenshots, and extracting information. The agent supports multi-step browsing sequences through conversational interaction.

## Installation

```bash
# Clone the repository
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/mcp_ai_agents/browser_mcp_agent

# Install Python dependencies
pip install -r requirements.txt

# Verify Node.js (required for Playwright MCP server)
node --version
npm --version

# Set API keys
export OPENAI_API_KEY=your-openai-api-key
# Or: export ANTHROPIC_API_KEY=your-anthropic-api-key
```

**Requirements:** Python 3.8+, Node.js + npm, OpenAI or Anthropic API key.

## Key Commands / Usage

### Start the Application
```bash
streamlit run main.py
```

### MCP Configuration (mcp_agent.config.yaml)
```yaml
mcp:
  servers:
    playwright:
      command: npx
      args: ["@anthropic/mcp-server-playwright"]
```

### Example Commands
```
# Basic Navigation
"Go to www.github.com"
"Go back to the previous page"

# Interaction
"Click on the login button"
"Fill in the search box with 'machine learning'"
"Scroll down to see more content"

# Content Extraction
"Summarize the main content of this page"
"Extract the navigation menu items"
"Take a screenshot of the hero section"

# Multi-step Tasks
"Go to Hacker News, find the top 5 stories, and summarize them"
"Navigate to Wikipedia, search for 'transformer model', and extract the key concepts"
```

### Secrets Configuration (mcp_agent.secrets.yaml)
```yaml
openai:
  api_key: "your-key"
anthropic:
  api_key: "your-key"
```

## Architecture
The agent uses a three-layer design: (1) **Streamlit Frontend** - provides the chat interface where users type natural language commands and see results with screenshots, (2) **MCP-Agent Orchestrator** - translates natural language into structured MCP tool calls using an LLM (GPT-4 or Claude), (3) **Playwright MCP Server** - executes browser actions (navigate, click, type, screenshot) in a headless Chromium instance. The MCP protocol standardizes communication between the LLM agent and browser tools, making it extensible with additional MCP servers.

## OS Agent Integration
Browser MCP Agent provides the web automation layer for OS-level agents. While desktop agents handle native applications, this tool handles everything in the browser. Integration patterns: (1) combine with a desktop agent for full computer control (native apps + browser), (2) chain with other MCP servers (filesystem, database) for end-to-end workflows, (3) use the Playwright MCP server standalone in any MCP-compatible agent framework. The MCP protocol ensures interoperability with Claude Code, Cursor, and custom agent systems.

## Source
- Repository: https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/mcp_ai_agents/browser_mcp_agent
- Parent Repo Stars: 93,998
- Language: Python
