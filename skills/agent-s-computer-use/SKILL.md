---
name: agent-s-computer-use
description: SOTA OS automation framework achieving 72.6% on OSWorld, first to surpass human performance
version: "1.0"
author: Simular AI
tags:
  - computer-use
  - os-agent
  - gui-automation
  - osworld
  - multimodal
  - cross-platform
tools:
  - gui-agents
  - agent-s3
---

# Agent S: Use Computer Like a Human

## Overview
Agent S is an open-source framework by Simular AI for autonomous computer interaction through an Agent-Computer Interface. Agent S3, the latest version, is the first to surpass human-level performance on OSWorld with a score of 72.6% (human baseline ~72%). It supports Windows, macOS, and Linux, and demonstrates strong zero-shot generalization across OSWorld, WindowsAgentArena (56.6%), and AndroidWorld (71.6%). The framework learns from past experiences and performs complex multi-step tasks autonomously.

## Installation

```bash
# Install from PyPI
pip install gui-agents

# Or from source for development
git clone https://github.com/simular-ai/Agent-S.git
cd Agent-S
pip install -e .

# Required: install tesseract for OCR
brew install tesseract       # macOS
apt install tesseract-ocr    # Linux
choco install tesseract      # Windows
```

**Requirements:** Python 3.10+, single monitor setup, API keys for LLM providers.

## Key Commands / Usage

### API Configuration
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
# Or use config file
```

### Running Agent S3
```python
from gui_agents import AgentS3

agent = AgentS3(
    model="claude-sonnet-4-5-20250929",
    platform="windows"  # or "macos", "linux"
)
agent.run("Book a flight from SFO to JFK on September 1st")
```

### CLI Usage
```bash
# Run with a task description
python -m gui_agents.run --task "Open browser and search for AI papers" --model claude-sonnet-4-5-20250929
```

### Cloud Option
Skip local setup entirely with [Simular Cloud](https://cloud.simular.ai/).

## Architecture
Agent S3 uses a three-stage pipeline: (1) Screen Understanding via accessibility tree + OCR + optional screenshots, (2) Action Planning using hierarchical task decomposition with experience-based retrieval, (3) Action Execution through platform-native APIs (UIA on Windows, AppleScript on macOS, AT-SPI on Linux). Behavior Best-of-N sampling generates multiple action rollouts and selects the best trajectory, boosting accuracy from 66% to 72.6%. The framework maintains an experience memory that improves performance across repeated tasks.

## OS Agent Integration
Agent S provides the highest-performing open-source computer-use agent available. It serves as the execution backbone for any system requiring autonomous GUI interaction. Key integration points: (1) cross-platform support means one agent works on all desktop OSes, (2) the `gui-agents` Python package allows programmatic embedding, (3) experience memory enables continuous improvement in production deployments. Papers accepted at ICLR 2025 (Best Paper at Agentic AI Workshop) and COLM 2025.

## Source
- Repository: https://github.com/simular-ai/Agent-S
- Stars: 9,730
- Language: Python
- Papers: [S3](https://arxiv.org/abs/2510.02250), [S2](https://arxiv.org/abs/2504.00906), [S1](https://arxiv.org/abs/2410.08164)
