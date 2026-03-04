---
name: windows-use-agent
description: Windows GUI automation agent using accessibility tree and vision, no CV models required
version: "1.0"
author: CursorTouch
tags:
  - windows
  - gui-automation
  - accessibility-tree
  - computer-use
  - os-agent
tools:
  - windows-use
  - Windows-MCP
---

# Windows Use Agent

## Overview
Windows-Use is a Windows-native GUI automation agent that interacts directly with the OS at the GUI layer. It uses the Windows accessibility tree (UIA) instead of computer vision models, enabling any LLM to perform desktop automation. The agent can open apps, click buttons, type text, execute shell commands, and capture UI state. A companion MCP server (Windows-MCP) exposes these capabilities over the Model Context Protocol for integration with Claude, Cursor, and other AI tools.

## Installation

```bash
# Install the core library
pip install windows-use

# Or clone and install from source
git clone https://github.com/CursorTouch/Windows-MCP.git
cd Windows-MCP
uv pip install -e .
```

**Requirements:**
- Python 3.12+
- Windows 10 or 11
- UV or pip

## Key Commands / Usage

### Standalone Agent

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from windows_use.agent import Agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
agent = Agent(llm=llm, use_vision=True)
result = agent.invoke(query="Open Notepad and type Hello World")
print(result.content)
```

### MCP Server Mode (for Claude/Cursor)

```json
{
  "mcpServers": {
    "windows-use": {
      "command": "python",
      "args": ["-m", "windows_mcp.server"]
    }
  }
}
```

**Core actions:** `click`, `type`, `scroll`, `screenshot`, `get_ui_tree`, `execute_shell`, `open_app`, `wait_for_element`

## Architecture
The agent reads the Windows UI Automation (UIA) accessibility tree to understand screen layout and interactive elements. Each action cycle: (1) capture accessibility tree snapshot, (2) optionally capture screenshot for vision models, (3) LLM decides next action based on tree + task, (4) execute action via Win32/UIA APIs. The MCP server wraps this loop for external AI tool integration. No dependency on specific vision models means any LLM (Gemini, Claude, GPT, Llama) can drive automation.

## OS Agent Integration
Windows-Use fills the Windows gap in the OS agent ecosystem. While most computer-use frameworks target macOS/Linux, this tool provides native Windows accessibility tree access. It can serve as the Windows execution backend for multi-platform agent orchestrators. The MCP server mode integrates directly with Claude Code, Cursor, and other MCP-compatible AI tools for natural language desktop control.

## Source
- Repository: https://github.com/CursorTouch/Windows-MCP
- Stars: 4,271
- Language: Python
- Also see: https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/single_agent_apps/windows_use_autonomous_agent
