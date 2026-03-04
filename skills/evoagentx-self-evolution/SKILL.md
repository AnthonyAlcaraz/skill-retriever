---
name: evoagentx-self-evolution
description: Self-evolving AI agent framework with automated workflow construction, evaluation, and optimization
version: "1.0"
author: EvoAgentX
tags:
  - self-evolution
  - agent-framework
  - workflow-optimization
  - multi-agent
  - auto-construction
  - human-in-the-loop
tools:
  - evoagentx
---

# EvoAgentX: Self-Evolving AI Agents

## Overview
EvoAgentX is an open-source framework for building, evaluating, and evolving LLM-based agents or agentic workflows in an automated, modular, and goal-driven manner. It moves beyond static prompt chaining: from a single prompt, it auto-constructs multi-agent workflows, evaluates them against task-specific criteria, and optimizes using self-evolving algorithms. Agents learn and improve through iterative feedback loops. Supports OpenAI, Claude, DeepSeek, Qwen, and local models via LiteLLM.

## Installation

```bash
# Install from PyPI
pip install evoagentx

# Or from source
git clone https://github.com/EvoAgentX/EvoAgentX.git
cd EvoAgentX
pip install -e .
```

### LLM Configuration
```python
from evoagentx.models import OpenAIModel

# Via environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Or via config
model = OpenAIModel(
    model_name="gpt-4o",
    api_key="your-key"
)
```

## Key Commands / Usage

### Auto-Construct a Workflow
```python
from evoagentx import EvoAgentX

eax = EvoAgentX(model="gpt-4o")

# Describe your goal, get a multi-agent workflow
workflow = eax.construct(
    goal="Research the latest AI papers on self-evolving agents, "
         "summarize the top 5, and draft a blog post"
)
workflow.run()
```

### Self-Evolution
```python
# Evolve the workflow based on evaluation results
evolved = eax.evolve(
    workflow=workflow,
    dataset=eval_dataset,
    iterations=5
)
```

### Human-in-the-Loop
```python
# Insert review checkpoints
workflow = eax.construct(
    goal="Draft quarterly report",
    hitl_checkpoints=["after_data_collection", "before_final_draft"]
)
```

### Built-in Tools
Search, code execution, browser interaction, file I/O, API calls, and more are included out of the box.

## Architecture
EvoAgentX has five core modules: (1) **Auto-Construction** - decomposes a goal into a structured multi-agent workflow with specialized agents for each subtask, (2) **Evaluation** - scores agent behavior using task-specific criteria with automatic evaluators, (3) **Self-Evolution Engine** - applies optimization algorithms (prompt evolution, workflow restructuring, agent replacement) based on evaluation scores, (4) **Memory** - both ephemeral (per-session) and persistent (cross-session) memory systems, (5) **Tool Library** - comprehensive built-in tools for real-world interaction. Supports plug-and-play model providers via LiteLLM, SiliconFlow, and OpenRouter.

## OS Agent Integration
EvoAgentX provides the meta-layer for agent systems: it does not just run agents, it evolves them. Integration patterns: (1) use as the workflow engine behind OS agents, letting it optimize how tasks are decomposed and executed, (2) plug in computer-use tools (Agent S, CUA) as EvoAgentX tools for GUI interaction, (3) leverage the self-evolution engine to automatically improve agent performance on repetitive tasks without manual prompt tuning. The framework paper and survey on self-evolving agents are published on arXiv.

## Source
- Repository: https://github.com/EvoAgentX/EvoAgentX
- Stars: 2,538
- Language: Python
- Papers: [Framework](https://arxiv.org/abs/2507.03616), [Survey](https://arxiv.org/abs/2508.07407)
