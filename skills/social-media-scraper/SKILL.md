---
name: social-media-scraper
description: Multi-agent social media scraping, podcast generation, and sentiment analysis pipeline (Beifong)
version: "1.0"
author: Shubhamsaboo / arun477
tags:
  - social-media
  - scraping
  - podcast-generation
  - sentiment-analysis
  - content-curation
  - browser-automation
tools:
  - beifong
  - browseruse
  - redis
---

# Beifong: Social Media Scraper + Podcast Generator

## Overview
Beifong is a comprehensive information and podcast generation system from the awesome-llm-apps collection. It manages trusted articles and social media sources, scrapes content from multiple platforms, performs AI-powered analysis and sentiment detection, and generates podcast scripts with audio. The complete pipeline runs from data collection through analysis to production of scripts and visuals. It supports scheduled feed collection, persistent browser sessions for social media login, and Slack integration.

## Installation

```bash
# Clone the repository
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/advanced_ai_agents/multi_agent_apps/ai_news_and_podcast_agents

# Install dependencies
pip install -r requirements.txt

# Install Redis (required)
# macOS: brew install redis
# Ubuntu: apt install redis-server
# Windows: Use Docker or WSL

# Configure environment
export OPENAI_API_KEY=your-key
# Optional: export ELEVENLABS_API_KEY=your-key
```

**Requirements:** Python 3.11+, Redis server, OpenAI API key, optional ElevenLabs for TTS.

## Key Commands / Usage

### Start the Application
```bash
# Start Redis first
redis-server

# Launch the web interface
python app.py
```

### Three Usage Methods
1. **Web UI** - Browse, curate sources, and generate podcasts via the dashboard
2. **CLI** - Run individual pipeline stages from the command line
3. **Scheduled** - Set up cron-based feed collection for continuous monitoring

### Social Media Monitoring
```python
# Supported platforms with persistent sessions
platforms = ["twitter", "linkedin", "reddit", "hackernews"]

# Configure feed sources
feeds = {
    "twitter": ["@elonmusk", "@OpenAI", "#AIagents"],
    "reddit": ["r/MachineLearning", "r/LocalLLaMA"],
    "hackernews": ["top", "new"]
}
```

### Content Processing Pipeline
1. **Collect** - Scrape content from configured sources using browser automation
2. **Analyze** - AI-powered summarization, sentiment analysis, topic extraction
3. **Curate** - Filter and rank content by relevance and quality
4. **Generate** - Create podcast scripts, audio (Kokoro/ElevenLabs TTS), and visuals
5. **Distribute** - Push to Slack or export as files

### Podcast Generation
```bash
# Generate podcast from collected content
python generate_podcast.py --topic "AI Agents" --sources today
```

## Architecture
Beifong uses a multi-agent pipeline: (1) **Content Processors** - pluggable scrapers for each platform (Twitter, LinkedIn, Reddit, HN) using browser automation with persistent login sessions, (2) **AI Agent** - orchestrates analysis with configurable tools for search, summarization, and classification, (3) **TTS Engine** - supports Kokoro (local, free) and ElevenLabs (cloud) for voice generation, (4) **Storage** - SQLite for metadata, Redis for task queues and caching, filesystem for media assets, (5) **Web Interface** - Flask-based dashboard for source management, content review, and podcast playback.

## OS Agent Integration
Beifong fills the content intelligence gap in OS agent systems. Integration patterns: (1) feed its scraped content into knowledge graphs or RAG systems for agent context, (2) use the browser automation module as a social media interaction layer for OS agents, (3) schedule automated monitoring to keep agents informed about trends and discussions, (4) combine with desktop agents for end-to-end content workflows (scrape, analyze, draft posts, publish). The persistent browser session approach handles authentication challenges that API-based scrapers cannot.

## Source
- Repository: https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/multi_agent_apps/ai_news_and_podcast_agents
- Parent Repo Stars: 93,998
- Language: Python
- Also see: https://github.com/arun477/beifong
