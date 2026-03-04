---
name: bytebot-desktop-agent
description: Self-hosted AI desktop agent with full virtual Ubuntu environment for autonomous task completion
version: "1.0"
author: bytebot-ai
tags:
  - desktop-agent
  - self-hosted
  - docker
  - computer-use
  - virtual-desktop
  - task-automation
tools:
  - bytebot
  - bytebot-api
---

# Bytebot: Open-Source AI Desktop Agent

## Overview
Bytebot is a self-hosted AI desktop agent that runs inside a containerized Linux desktop environment. Unlike browser-only agents, Bytebot gets its own full virtual computer with a real desktop, filesystem, and installed applications. It can use any application (browsers, email clients, office tools, IDEs), download and organize files, log into websites, read PDFs and spreadsheets, and complete complex multi-step workflows across different programs. Think of it as a virtual employee with their own computer.

## Installation

### Option 1: Railway (Easiest)
Click deploy at [Railway](https://railway.com/deploy/bytebot) and add your AI provider API key.

### Option 2: Docker Compose
```bash
git clone https://github.com/bytebot-ai/bytebot.git
cd bytebot

# Add your AI provider key
echo "ANTHROPIC_API_KEY=sk-ant-..." > docker/.env
# Or: echo "OPENAI_API_KEY=sk-..." > docker/.env
# Or: echo "GEMINI_API_KEY=..." > docker/.env

docker-compose -f docker/docker-compose.yml up -d

# Open http://localhost:9992
```

## Key Commands / Usage

### Web Interface
Navigate to `http://localhost:9992` after deployment. Create tasks in natural language:
- "Download all invoices from vendor portals and organize them into a folder"
- "Read this PDF and extract the key financial data into a spreadsheet"
- "Set up a development environment with Node.js and create a new project"

### REST API
```bash
# Create a task
curl -X POST http://localhost:9992/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "Open Firefox and search for latest AI papers"}'

# Get task status
curl http://localhost:9992/api/tasks/{task_id}

# Upload files for processing
curl -X POST http://localhost:9992/api/tasks/{task_id}/files \
  -F "file=@invoice.pdf"
```

### File Upload
Drop files directly onto tasks in the web UI for Bytebot to process (PDFs, spreadsheets, documents).

## Architecture
Bytebot has four integrated components: (1) **Virtual Desktop** - a full Ubuntu Linux environment with pre-installed apps (Firefox, LibreOffice, VS Code, terminal), (2) **AI Agent** - powered by Claude, GPT-4, or Gemini, understands tasks and controls the desktop via screenshots + mouse/keyboard, (3) **Task Interface** - web UI for creating tasks, uploading files, and watching the agent work in real-time, (4) **REST APIs** - programmatic endpoints for task creation and desktop control. The agent loop captures screenshots, reasons about the current state, and executes actions through X11 input injection.

## OS Agent Integration
Bytebot provides the most turnkey desktop agent experience: deploy one Docker container and get a complete AI-controlled desktop. Key integration patterns: (1) use the REST API to create tasks from other automation systems, (2) pre-install custom applications in the Docker image for domain-specific workflows, (3) mount volumes for persistent file storage across sessions, (4) connect to VNC (port 5900) for visual monitoring. The containerized approach provides natural isolation and security, making it suitable for production workloads involving untrusted web interactions.

## Source
- Repository: https://github.com/bytebot-ai/bytebot
- Stars: 10,392
- Language: TypeScript
- Documentation: https://docs.bytebot.ai
