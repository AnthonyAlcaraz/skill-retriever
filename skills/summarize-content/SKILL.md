---
name: summarize
description: Summarize any URL, YouTube video, podcast, file, or media via CLI and Chrome extension
version: "0.10"
author: steipete
tags:
  - summarization
  - youtube
  - podcast
  - chrome-extension
  - cli
  - media
  - transcription
tools:
  - summarize (CLI)
  - summarize daemon
  - Chrome Side Panel extension
  - Firefox Sidebar extension
---

# Summarize - Content Summarization Engine

## Overview

Summarize provides fast summaries from URLs, files, and media through both a CLI and a Chrome Side Panel / Firefox Sidebar extension. It handles web pages, PDFs, images, audio/video, YouTube videos (with slide extraction and OCR), podcasts, and RSS feeds. The CLI supports streaming Markdown output with metrics, cache-aware status, and cost estimates. The browser extension provides one-click summarization in a side panel with a streaming chat interface.

Key capabilities: transcript-first media flow (published transcripts when available, Whisper fallback when not), YouTube slide extraction with OCR and timestamped cards, support for local/paid/free models (OpenAI-compatible local endpoints, paid providers, OpenRouter free preset), and multiple output modes (Markdown, JSON diagnostics, extract-only, metrics).

## Installation

```bash
# npm (global, cross-platform)
npm install -g @steipete/summarize

# Homebrew (macOS)
brew install steipete/tap/summarize

# npx (no install)
npx -y @steipete/summarize "https://example.com"
```

Chrome extension: [Chrome Web Store - Summarize Side Panel](https://chromewebstore.google.com/detail/summarize/cejgnmmhbbpdmjnfppjdfkocebngehfg)

## Key Commands / Usage

```bash
# Summarize a web page
summarize "https://example.com/article"

# Summarize a YouTube video
summarize "https://www.youtube.com/watch?v=..."

# Summarize a local file
summarize ~/Documents/report.pdf

# Summarize with specific model
summarize --model gpt-5.1 "https://example.com"

# Extract only (no summary, just content)
summarize --extract "https://example.com"

# JSON output with metrics
summarize --json --metrics "https://example.com"

# Force summary even for short content
summarize --force-summary "https://example.com/short-page"

# Daemon for Chrome extension
summarize daemon install --token <TOKEN>
```

For YouTube, select "Video + Slides" in the extension to get screenshots + OCR + transcript cards with timestamped seek.

## Architecture

The CLI handles content type detection and routes to appropriate extractors: web pages go through Readability for Markdown conversion, YouTube through yt-dlp for transcripts and ffmpeg for slide extraction, audio/video through Whisper for transcription, PDFs through dedicated parsers. The streaming Markdown renderer shows output progressively with metrics.

The Chrome extension communicates with a localhost daemon service that handles the heavy extraction work (yt-dlp, ffmpeg, OCR, transcription). The daemon autostarts via launchd (macOS), systemd (Linux), or Scheduled Task (Windows) and requires a shared token for authentication.

## Integration with OS Agent

Summarize serves as the "content ingestion" tool in an OS agent stack. When an agent needs to understand a web page, digest a YouTube tutorial, extract information from a PDF, or process a podcast episode, Summarize handles the extraction and condensation. The CLI mode integrates directly into agent workflows, while the daemon mode supports browser-integrated use. Combined with Oracle for multi-model analysis and Conduit MCP for file operations, Summarize completes the information gathering pipeline.

## Source

- Repository: https://github.com/steipete/summarize
- Stars: 1646
- Language: TypeScript
