---
name: anthropic-computer-use
description: Official Anthropic computer-use demo with Docker, Streamlit UI, and support for Claude 4 models
version: "1.0"
author: Anthropic
tags:
  - computer-use
  - anthropic
  - claude
  - docker
  - desktop-control
  - official
tools:
  - computer-use-demo
  - str_replace_based_edit_tool
  - bash_tool
  - computer_tool
---

# Anthropic Computer Use Demo

## Overview
The official Anthropic reference implementation for Claude's computer use capabilities. It provides a Docker container with a full Linux desktop environment that Claude can control: moving the mouse, clicking, typing, taking screenshots, and running commands. Supports Claude Opus 4.5, Claude Sonnet 4.5, Claude Sonnet 4, Claude Opus 4, and Claude Haiku 4.5. Includes a Streamlit web interface for interactive sessions and supports Anthropic API, AWS Bedrock, and Google Vertex as providers.

## Installation

### Claude API (Direct)
```bash
export ANTHROPIC_API_KEY=your_api_key

docker run \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -v $HOME/.anthropic:/home/computeruse/.anthropic \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
```

### AWS Bedrock
```bash
docker run \
    -e API_PROVIDER=bedrock \
    -e AWS_PROFILE=$AWS_PROFILE \
    -e AWS_REGION=us-west-2 \
    -v $HOME/.aws:/home/computeruse/.aws \
    -v $HOME/.anthropic:/home/computeruse/.anthropic \
    -p 5900:5900 -p 8501:8501 -p 6080:6080 -p 8080:8080 \
    -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
```

### Google Vertex
```bash
docker build . -t computer-use-demo
gcloud auth application-default login
docker run \
    -e API_PROVIDER=vertex \
    -e CLOUD_ML_REGION=$VERTEX_REGION \
    -e ANTHROPIC_VERTEX_PROJECT_ID=$VERTEX_PROJECT_ID \
    -v $HOME/.config/gcloud/application_default_credentials.json:/home/computeruse/.config/gcloud/application_default_credentials.json \
    -p 5900:5900 -p 8501:8501 -p 6080:6080 -p 8080:8080 \
    -it computer-use-demo
```

## Key Commands / Usage

### Access Points (after Docker start)
- **Streamlit UI**: http://localhost:8501 (main interface)
- **VNC Viewer**: http://localhost:6080 (noVNC web client)
- **VNC Direct**: vnc://localhost:5900
- **API**: http://localhost:8080

### Anthropic-Defined Tools
- **computer_tool**: mouse movement, clicking, typing, screenshots, scrolling
- **str_replace_based_edit_tool**: file editing with search-and-replace
- **bash_tool**: execute shell commands

### Supported Models
- Claude Opus 4.5 (`claude-opus-4-5-20251101`)
- Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- Claude Opus 4 (`claude-opus-4-20250514`)
- Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)

## Architecture
The demo packages three components in one Docker container: (1) **Agent Loop** - Python process that sends screenshots to Claude, receives tool calls, and executes them in a loop until the task is complete, (2) **Linux Desktop** - Xfce4 desktop environment with pre-installed apps (Firefox, text editor, terminal), accessible via VNC, (3) **Streamlit App** - web interface for entering tasks, viewing the conversation, and watching the agent work. The agent uses Claude's native `computer_use_20251124` tool specification with zoom actions. Single-session design; must restart between sessions.

## OS Agent Integration
This is the canonical reference for building computer-use agents with Claude. Integration patterns: (1) fork and customize the Docker image with your own applications and tools, (2) extract the agent loop code for embedding in your own systems, (3) use as a testing environment for evaluating Claude's computer-use capabilities, (4) study the tool implementations as the official spec for computer_tool, bash_tool, and str_replace_based_edit_tool. Also includes the Browser Tools API demo for Playwright-based web automation as an alternative to screenshot-based browsing.

## Source
- Repository: https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo
- Stars: 13,908
- Language: Python
- Documentation: https://docs.claude.com/en/docs/computer-use
