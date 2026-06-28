# LLM Wiki — Personal AI Knowledge Base Agent

> Inspired by Andrej Karpathy's LLM Wiki pattern. Built from scratch using
> NVIDIA NIM's free API — no paid keys needed.

A personal knowledge base that compounds over time. Drop in articles, papers,
and notes — an AI agent synthesises them into a structured, interlinked wiki.
Query it, detect gaps, and rank sources by contribution.

---

## What Makes It Different From RAG

Most LLM + document setups use RAG — retrieve chunks at query time, generate
an answer. Knowledge never accumulates.

LLM Wiki is different. When you add a new source, the agent integrates it into
the existing wiki — updating pages, adding cross-links, noting contradictions.
The wiki compounds. Each new source makes every existing page richer.

---

## Features

- **Web clipper** — paste a URL or drop a PDF, content ingested automatically
- **Index-first compilation** — prompts stay lean even as the wiki grows to 100+ pages
- **Agent loop** — agent checks its own output and fills missing linked pages automatically
- **Conflict detection** — flags when new sources contradict existing wiki content
- **Watcher** — drop files into `sources/` and compilation happens in the background
- **Web UI** — upload content, chat with your wiki, view insights from the browser
- **Gap detector** — surfaces missing topics and orphan links in your knowledge base
- **Source ranker** — scores which sources contributed most to your wiki

---

## Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.11+ |
| LLM API | NVIDIA NIM (free tier) — [build.nvidia.com](https://build.nvidia.com) |
| Models | Llama 3.3 70B · DeepSeek V4 via OpenAI-compatible endpoint |
| Web UI | Flask |
| Web scraping | Trafilatura |
| PDF extraction | PyMuPDF |
| File watching | Watchdog |
| Graph view | Obsidian (optional) |

---

## Project Structure

```
llm-wiki/
├── compiler.py         Core agent — index-first + loop + conflict detection
├── clipper.py          URL and PDF ingestion into sources/
├── watcher.py          Auto-compiles on file drop
├── app.py              Web UI — upload, chat, insights tab
├── insights.py         Gap detector + source ranker (CLI)
├── AGENT.md            Schema governing how the agent writes wiki pages
├── LLM_WIKI_GUIDE.md   Complete usage guide with all terminal commands
├── requirements.txt
└── README.md
```

Runtime folders (git-ignored, created on first run):
```
sources/    Your raw inputs (articles, PDFs, notes)
wiki/       Agent output — structured, interlinked markdown pages
```

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/llm-wiki.git
cd llm-wiki
```

**2. Install dependencies**
```bash
pip3 install -r requirements.txt
```

**3. Get a free API key**

Sign up at [build.nvidia.com](https://build.nvidia.com) — no credit card needed.

```bash
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx

# To make it permanent on Mac:
echo 'export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx' >> ~/.zprofile
source ~/.zprofile
```

**4. Create runtime folders**
```bash
mkdir sources wiki
```

**5. Add a source and compile**
```bash
echo "The Transformer was introduced in 2017 by Vaswani et al..." > sources/test.txt
python3 compiler.py
```

**6. Query your wiki**
```bash
python3 compiler.py --query "What is a Transformer?"
```

**7. Or launch the web UI**
```bash
python3 app.py
# Open http://localhost:5000
```

---

## Usage Reference

### compiler.py — Core Agent

```bash
python3 compiler.py                        # compile all new sources
python3 compiler.py --file notes.md        # compile a specific file
python3 compiler.py --all                  # recompile everything from scratch
python3 compiler.py --query "What is X?"  # ask a question against your wiki
```

### clipper.py — URL & PDF Ingestion

```bash
python3 clipper.py https://some-article.com          # clip a web article
python3 clipper.py ~/Downloads/paper.pdf             # ingest a PDF
python3 clipper.py https://some-article.com --no-compile  # save without compiling
python3 clipper.py --list                            # show all clipped sources
```

### watcher.py — Auto Compiler

```bash
python3 watcher.py          # watch sources/ and auto-compile on file drop
python3 watcher.py --once   # process pending files and exit
```

### app.py — Web UI

```bash
python3 app.py
# Open http://localhost:5000
```

- **Left panel** — paste a URL or drop a PDF to add content
- **Chat tab** — multi-turn conversation against your wiki with citations
- **Insights tab** — gap detection and source rankings

### insights.py — Wiki Health

```bash
python3 insights.py          # run gap detection + source ranking
python3 insights.py --gaps   # gap detection only
python3 insights.py --rank   # source ranking only
```

---

## How the Agent Works

```
New source dropped in
        ↓
Step 1  Load INDEX.md only → ask model which existing pages are relevant
        ↓
Step 2  Load only relevant pages → compile source into wiki (not everything)
        ↓
Step 3  Check for conflicts with existing pages → log any contradictions found
        ↓
Step 4  Scan own output for [[orphan links]] → auto-create stub pages
        ↓
        Wiki updated · compounded · self-healed
```

This index-first pattern keeps prompts small even as the wiki grows to
hundreds of pages — only relevant context is loaded, not the entire wiki.

---

## Web UI Overview

| Panel | What it does |
|-------|-------------|
| Left — Add Content | Paste URL or drop PDF → live compile log streams in |
| Right — Chat | Ask questions, get answers with wiki page citations |
| Right — Insights | Missing topics, orphan links, source rankings with score bars |

Conflict warnings appear as a yellow badge in the header when a new source
contradicts existing wiki content. All conflicts logged to `wiki/CONFLICTS.md`.

---

## Troubleshooting

**API key not found**
```bash
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
```

**504 timeout error**
The free tier is occasionally overloaded. The compiler auto-retries 3 times.
Switch to a faster model if it keeps failing:
```bash
# In compiler.py, change MODEL to:
MODEL = "deepseek-ai/deepseek-v4-flash"
```

**Module not found**
```bash
pip3 install -r requirements.txt
```

**URL clipping fails**
The page likely requires JavaScript. Copy the article text manually into a
`.txt` file in `sources/` instead.

**Scanned PDF fails**
PyMuPDF handles text-based PDFs only. Scanned image PDFs are not yet supported.

---

## Changing the Model

The default model is `meta/llama-3.3-70b-instruct`. To switch, update the
`MODEL` variable at the top of `compiler.py`, `app.py`, and `insights.py`:

```python
MODEL = "deepseek-ai/deepseek-v4-flash"    # faster
MODEL = "deepseek-ai/deepseek-v4-pro"      # more powerful, slower
MODEL = "meta/llama-3.1-70b-instruct"      # alternative
```

To list all models available on your key:
```bash
python3 -c "
from openai import OpenAI; import os
c = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key=os.environ['NVIDIA_API_KEY'])
[print(m.id) for m in c.models.list().data]
"
```

---

## Inspired By

Andrej Karpathy's [llm-wiki.md](https://gist.github.com/karpathy) — the idea
that an LLM should act as a wiki maintainer that compounds knowledge over time,
not just a retrieval engine that answers questions from scratch on every query.

---

*Built with NVIDIA NIM free tier · No paid API keys required*
