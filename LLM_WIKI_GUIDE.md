# LLM Wiki — Complete Guide

Your personal AI-powered knowledge base. Drop sources in, the agent synthesises
them into a structured wiki. Query it, track gaps, rank sources. Everything runs
locally on your Mac using NVIDIA's free API.

---

## What You Built

```
llm-wiki/
├── compiler.py     Core agent — reads sources, writes wiki pages
├── clipper.py      Fetch URLs or ingest PDFs into sources/
├── watcher.py      Auto-compiles whenever you drop a file in sources/
├── app.py          Web UI — upload content, chat with wiki, view insights
├── insights.py     Gap detector + source ranker (also runs via UI)
├── AGENT.md        Schema that governs how the agent writes the wiki
├── sources/        Your raw inputs (articles, PDFs, notes)
└── wiki/           Agent output — structured markdown pages
    ├── INDEX.md
    ├── CONFLICTS.md
    └── *.md        One page per concept
```

---

## One-Time Setup

```bash
# 1. Navigate to your project
cd ~/Documents/llm-wiki

# 2. Install all dependencies
pip3 install openai trafilatura pymupdf flask watchdog

# 3. Set your NVIDIA API key (get it free at build.nvidia.com)
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx

# To make the key permanent across Terminal sessions:
echo 'export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx' >> ~/.zprofile
source ~/.zprofile
```

---

## Every Time You Open Terminal

```bash
# Always start here
cd ~/Documents/llm-wiki

# Make sure your API key is set
echo $NVIDIA_API_KEY   # should print your nvapi-... key
```

---

## compiler.py — Core Agent

The brain of the system. Reads source files and compiles them into wiki pages.

```bash
# Compile all new sources (files not yet compiled)
python3 compiler.py

# Compile a specific file
python3 compiler.py --file my_notes.md

# Recompile everything from scratch
python3 compiler.py --all

# Ask a question against your wiki (terminal mode)
python3 compiler.py --query "What is self-attention?"
python3 compiler.py --query "How does GPT differ from BERT?"
```

**What happens when you compile:**
1. Checks INDEX.md to find relevant existing pages
2. Loads only those pages (not everything) to keep prompts lean
3. Detects conflicts between new source and existing wiki
4. Writes/updates wiki pages and INDEX.md
5. Scans its own output for missing linked pages and creates stubs

**Output files written to:** `wiki/`

---

## clipper.py — URL & PDF Ingestion

Fetch any article or ingest any PDF directly into your sources folder, then
compiles automatically.

```bash
# Clip a web article
python3 clipper.py https://some-article.com

# Ingest a PDF (give the full path or filename if already in llm-wiki/)
python3 clipper.py ~/Downloads/attention-is-all-you-need.pdf
python3 clipper.py paper.pdf

# Save to sources/ but don't compile yet
python3 clipper.py https://some-article.com --no-compile
python3 clipper.py ~/Downloads/paper.pdf --no-compile

# Then compile everything pending at once
python3 compiler.py

# See all previously clipped sources
python3 clipper.py --list
```

**Supported inputs:**
- Any article URL that serves plain HTML (blogs, Wikipedia, Substack, Medium, etc.)
- Text-based PDFs (research papers, reports, books)
- Note: JavaScript-rendered pages and scanned PDFs are not supported

---

## watcher.py — Auto Compiler

Runs in the background and automatically compiles any file dropped into sources/.
No commands needed after starting it.

```bash
# Start watching (keep this running in a dedicated terminal tab)
python3 watcher.py

# Process any pending files right now, then exit
python3 watcher.py --once

# Stop watching
Ctrl+C
```

**Workflow with watcher running:**
1. Open a terminal tab, run `python3 watcher.py`
2. Leave it running in the background
3. Drop any .txt, .md, or .pdf into sources/ from Finder
4. Watch it compile automatically in the watcher tab
5. That's it — wiki updates without you doing anything else

---

## app.py — Web UI

Full browser interface. Upload content, chat with your wiki, view insights.

```bash
# Start the web server
python3 app.py

# Then open in your browser:
# http://localhost:5000

# Stop the server
Ctrl+C
```

**Left panel — Add Content:**
- Paste a URL → click "Clip & Compile" → compile log streams live
- Drop or select a PDF → same automatic flow
- Compile log shows each step in real time
- Conflicts appear as a yellow warning in the header if detected

**Right panel — Chat tab:**
- Type any question → get answers sourced from your wiki
- Multi-turn: ask follow-up questions, it remembers the conversation
- Citation tags show which wiki pages each answer came from
- Click hint chips to get started quickly
- "Clear chat" resets the conversation

**Right panel — Insights tab:**
- Click "Run Analysis" → takes ~30 seconds
- Shows: missing topics, orphan links, thin pages, source rankings
- Source rankings scored by how many wiki pages cite each source
- Re-run anytime after adding new content

---

## insights.py — Gap Detector & Source Ranker

Can also be run directly from terminal for a detailed terminal report.

```bash
# Run both gap detection and source ranking
python3 insights.py

# Gap detection only — what is your wiki missing?
python3 insights.py --gaps

# Source ranking only — which sources contributed most?
python3 insights.py --rank
```

**Gap detector finds:**
- Missing topics the wiki should have but doesn't
- Orphan links — concepts referenced but no page exists
- Thin pages — under 120 words, need more content
- Suggested reading — specific sources to fill the gaps

**Source ranker scores each source by:**
- How many wiki pages cite it
- How many new concepts it introduced
- Word count contribution
- Outputs ranked list with 🥇🥈🥉 medals + qualitative insight paragraph

---

## Obsidian — Visual Graph View

Point Obsidian at your wiki/ folder to see your knowledge as a visual graph.

```
1. Download Obsidian free from obsidian.md
2. Open Obsidian → "Open folder as vault"
3. Select: ~/Documents/llm-wiki/wiki/
4. Press Cmd+G to open graph view
```

The [[wiki-links]] the agent writes become clickable connections.
Clusters = dense knowledge. Isolated nodes = gaps to fill.
Reopen Obsidian after each compile to see new pages appear.

---

## Typical Daily Workflow

**Option A — Terminal workflow:**
```bash
cd ~/Documents/llm-wiki

# Add content
python3 clipper.py https://article-you-found.com
python3 clipper.py ~/Downloads/paper.pdf

# Query
python3 compiler.py --query "What did I learn about X?"

# Check health weekly
python3 insights.py
```

**Option B — UI workflow:**
```bash
cd ~/Documents/llm-wiki
python3 app.py
# Open http://localhost:5000
# Use the left panel to add URLs and PDFs
# Use the Chat tab to query
# Use the Insights tab to check gaps
```

**Option C — Fully automatic:**
```bash
cd ~/Documents/llm-wiki
python3 watcher.py   # leave running in background tab
# Now just drag files into sources/ from Finder
# Everything else is automatic
```

---

## Conflict Detection

When a new source contradicts existing wiki content, the compiler flags it.

- Terminal: printed as ⚠️ warnings during compile
- Web UI: yellow warning badge appears in the header
- All conflicts logged permanently to: `wiki/CONFLICTS.md`

Review `CONFLICTS.md` to decide which source to trust and update wiki pages manually if needed.

---

## Changing the Model

The default model is `meta/llama-3.3-70b-instruct`. To switch:

```bash
# In compiler.py, clipper.py, insights.py, and app.py, find this line:
MODEL = "meta/llama-3.3-70b-instruct"

# Replace with any model from your NVIDIA catalog:
MODEL = "deepseek-ai/deepseek-v4-flash"    # faster
MODEL = "deepseek-ai/deepseek-v4-pro"      # more powerful, slower
MODEL = "meta/llama-3.1-70b-instruct"      # alternative

# To see all available models:
python3 -c "
from openai import OpenAI; import os
c = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key=os.environ['NVIDIA_API_KEY'])
[print(m.id) for m in c.models.list().data]
"
```

---

## Troubleshooting

**API key not found:**
```bash
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
```

**504 timeout error:**
The free tier is busy. The compiler retries automatically 3 times.
If it keeps failing, switch to a smaller/faster model (see above).

**Module not found error:**
```bash
pip3 install openai trafilatura pymupdf flask watchdog
```

**Wrong Python version:**
```bash
python3 --version   # need 3.8 or higher
```

**URL clipping fails:**
The page likely requires JavaScript to render.
Copy the article text manually into a .txt file in sources/ instead.

**Scanned PDF fails:**
PyMuPDF only handles text-based PDFs.
Scanned/image PDFs need OCR — not yet supported.

**File already compiled, want to redo it:**
```bash
# Delete the compiled log entry and rerun
python3 compiler.py --all   # recompiles everything
```

---

## Key Files to Know

| File | What it is | Edit it? |
|------|-----------|----------|
| `AGENT.md` | Schema governing wiki style | Yes — customise how the agent writes |
| `wiki/INDEX.md` | Master index of all pages | No — agent maintains this |
| `wiki/CONFLICTS.md` | Log of detected conflicts | Read only — for your review |
| `.compiled` | List of compiled source filenames | No — tracks state automatically |
| `.clipped` | Log of clipped URLs | No — tracks state automatically |

---

*Built with NVIDIA NIM (free tier) · DeepSeek / Llama via OpenAI-compatible API*
*Inspired by Andrej Karpathy's LLM Wiki pattern*
