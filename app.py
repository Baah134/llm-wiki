"""
app.py — LLM Wiki Web UI
Query your wiki and add content from the browser.

Usage:
    python app.py
    Open: http://localhost:5000
"""

import os
import sys
import re
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

try:
    from openai import OpenAI
except ImportError:
    print("❌  openai not installed. Run: pip3 install openai")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
WIKI_DIR   = BASE_DIR / "wiki"
SOURCES_DIR= BASE_DIR / "sources"
INDEX_FILE = WIKI_DIR / "INDEX.md"
MODEL      = "meta/llama-3.3-70b-instruct"

# ── NVIDIA NIM client ─────────────────────────────────────────────────────────

def get_client():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("❌  NVIDIA_API_KEY not set.")
        print("    Export it first: export NVIDIA_API_KEY=nvapi-xxxx")
        sys.exit(1)
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

client = get_client()

# ── Wiki helpers ──────────────────────────────────────────────────────────────

def load_wiki_index() -> str:
    return INDEX_FILE.read_text(encoding="utf-8") if INDEX_FILE.exists() else "_(Wiki is empty.)_"

def load_all_wiki_pages() -> str:
    pages = []
    for f in sorted(WIKI_DIR.glob("*.md")):
        if f.name in ("INDEX.md", "CONFLICTS.md"):
            continue
        pages.append(f"### FILE: {f.name}\n\n{f.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(pages) if pages else "_(No wiki pages yet.)_"

def get_wiki_stats() -> dict:
    pages = [f for f in WIKI_DIR.glob("*.md") if f.name not in ("INDEX.md", "CONFLICTS.md", "_raw_output.md")]
    conflicts_file = WIKI_DIR / "CONFLICTS.md"
    conflict_count = 0
    if conflicts_file.exists():
        conflict_count = conflicts_file.read_text().count("## ")
    return {
        "page_count": len(pages),
        "pages": [f.stem for f in sorted(pages)],
        "conflict_count": conflict_count,
    }

def get_conflicts() -> str:
    f = WIKI_DIR / "CONFLICTS.md"
    return f.read_text(encoding="utf-8") if f.exists() else ""

def call_api(messages: list, max_tokens: int = 4096) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.4,
                top_p=0.95,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(10)
            else:
                raise

def query_wiki(question: str, history: list = []) -> dict:
    wiki_index    = load_wiki_index()
    existing_wiki = load_all_wiki_pages()

    system_prompt = f"""You are a knowledgeable assistant with access to a personal wiki.
Answer questions using ONLY the information in the wiki pages below.
If the answer isn't in the wiki, say so clearly.
At the end of your answer, list the wiki pages you drew from under a "Sources:" heading.

## Wiki Index
{wiki_index}

## Wiki Pages
{existing_wiki}""".strip()

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})

    answer = call_api(messages)

    citations = []
    if "Sources:" in answer:
        parts = answer.split("Sources:", 1)
        answer_text = parts[0].strip()
        sources_text = parts[1].strip()
        citations = [
            line.strip().lstrip("-•* ").strip()
            for line in sources_text.splitlines()
            if line.strip() and line.strip() not in ("-", "•", "*")
        ]
    else:
        answer_text = answer

    return {"answer": answer_text, "citations": citations}

# ── Ingestion helpers ─────────────────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path   = parsed.path.strip("/").replace("/", "_")
    slug   = f"{domain}_{path}" if path else domain
    slug   = re.sub(r"[^\w\-]", "_", slug)
    slug   = re.sub(r"_+", "_", slug).strip("_")[:80]
    return f"{datetime.now().strftime('%Y%m%d')}_{slug}.txt"

def ingest_url(url: str) -> tuple[str, str]:
    """Fetch and extract article text. Returns (filename, status_message)."""
    try:
        import trafilatura
    except ImportError:
        return "", "trafilatura not installed. Run: pip3 install trafilatura"

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return "", "Could not fetch the URL. It may require JavaScript or be unavailable."

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else url

    if not text:
        return "", "Could not extract text. The page may be JavaScript-rendered or paywalled."

    SOURCES_DIR.mkdir(exist_ok=True)
    filename = url_to_filename(url)
    filepath = SOURCES_DIR / filename
    content = f"# {title}\n\n**Source URL:** {url}\n**Clipped:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n{text}"
    filepath.write_text(content, encoding="utf-8")
    return filename, f"Saved: {filename} ({len(text.split()):,} words extracted)"

def ingest_pdf(file_data: bytes, original_name: str) -> tuple[str, str]:
    """Extract text from PDF bytes. Returns (filename, status_message)."""
    try:
        import fitz
    except ImportError:
        return "", "PyMuPDF not installed. Run: pip3 install pymupdf"

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        pages_text = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
    finally:
        os.unlink(tmp_path)

    if not pages_text:
        return "", "Could not extract text. This may be a scanned/image PDF."

    full_text = "\n\n".join(pages_text)
    title = Path(original_name).stem.replace("_", " ").replace("-", " ").title()

    SOURCES_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^\w\-]", "_", Path(original_name).stem)[:80]
    filename = f"{datetime.now().strftime('%Y%m%d')}_{slug}.txt"
    filepath = SOURCES_DIR / filename
    content = f"# {title}\n\n**Source:** {original_name} (PDF)\n**Ingested:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n{full_text}"
    filepath.write_text(content, encoding="utf-8")
    return filename, f"Saved: {filename} ({len(full_text.split()):,} words extracted from {len(pages_text)} pages)"

# ── Compile job runner (SSE streaming) ───────────────────────────────────────

def run_compile_stream(filename: str):
    """
    Generator that runs compiler.py and streams log lines as SSE events.
    """
    import subprocess
    cmd = [sys.executable, str(BASE_DIR / "compiler.py"), "--file", filename]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in process.stdout:
        line = line.rstrip()
        if line:
            yield f"data: {json.dumps({'log': line})}\n\n"
    process.wait()
    status = "done" if process.returncode == 0 else "error"
    yield f"data: {json.dumps({'status': status})}\n\n"

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Wiki</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #f7f8fc;
    --surface:   #ffffff;
    --surface2:  #eef0f8;
    --border:    #dde0ee;
    --accent:    #5b4ecc;
    --accent-dim:#ebe8ff;
    --text:      #1a1c2e;
    --muted:     #8a8fa8;
    --green:     #16a34a;
    --yellow:    #d97706;
    --red:       #dc2626;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }

  html, body { height: 100%; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  .logo { font-family: var(--mono); font-size: 13px; color: var(--accent); letter-spacing: 0.08em; }
  .logo span { color: var(--muted); }

  #stats { font-family: var(--mono); font-size: 11px; color: var(--muted); display: flex; gap: 16px; }
  #stats b { color: var(--green); }
  #conflict-stat { color: var(--yellow); cursor: pointer; }
  #conflict-stat:hover { color: var(--yellow); text-decoration: underline; }

  /* ── Layout ── */
  .layout {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Left panel — Add Content ── */
  .panel-left {
    width: 300px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }

  .panel-title {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    padding: 16px 20px 10px;
    border-bottom: 1px solid var(--border);
  }

  .add-section { padding: 16px 20px; border-bottom: 1px solid var(--border); }

  .add-section h3 {
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 10px;
  }

  .url-input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 12px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 300;
    outline: none;
    margin-bottom: 8px;
    transition: border-color 0.15s;
  }
  .url-input:focus { border-color: var(--accent); }
  .url-input::placeholder { color: var(--muted); }

  .btn {
    width: 100%;
    padding: 8px 12px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 500;
    transition: opacity 0.15s;
  }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover:not(:disabled) { opacity: 0.85; }
  .btn-secondary {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    margin-top: 8px;
  }
  .btn-secondary:hover:not(:disabled) { border-color: var(--accent); }

  /* Drop zone */
  .drop-zone {
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: var(--surface2);
  }
  .drop-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .drop-icon { font-size: 22px; margin-bottom: 6px; }
  .drop-label { font-size: 11px; color: var(--muted); line-height: 1.5; }
  .drop-label b { color: var(--text); }

  /* Log output */
  .log-area {
    padding: 12px 20px;
    flex: 1;
  }

  .log-title {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }

  #compile-log {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    line-height: 1.7;
    min-height: 60px;
    white-space: pre-wrap;
    word-break: break-all;
  }

  #compile-log .log-done { color: var(--green); }
  #compile-log .log-warn { color: var(--yellow); }
  #compile-log .log-err  { color: var(--red); }

  /* Conflicts panel */
  #conflicts-panel {
    display: none;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  #conflicts-panel .conflict-title {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--yellow);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }
  #conflicts-content {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.6;
    white-space: pre-wrap;
    max-height: 200px;
    overflow-y: auto;
  }

  /* ── Right panel — Chat ── */
  .panel-right {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  #history {
    flex: 1;
    overflow-y: auto;
    padding: 28px 32px 16px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .turn { display: flex; flex-direction: column; gap: 10px; }

  .question {
    align-self: flex-end;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    font-size: 13px;
    max-width: 80%;
    line-height: 1.5;
  }

  .answer-wrap { align-self: flex-start; max-width: 92%; display: flex; flex-direction: column; gap: 6px; }

  .answer {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px 12px 12px 12px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.7;
    white-space: pre-wrap;
  }

  .citations { display: flex; flex-wrap: wrap; gap: 5px; padding-left: 2px; }

  .citation-tag {
    font-family: var(--mono);
    font-size: 10px;
    background: transparent;
    border: 1px solid var(--accent-dim);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 999px;
  }

  /* Empty state */
  #empty {
    text-align: center;
    padding: 80px 32px;
    color: var(--muted);
  }
  #empty h2 { font-family: var(--mono); font-size: 16px; color: var(--text); margin-bottom: 8px; font-weight: 500; }
  #empty p { font-size: 12px; line-height: 1.6; }
  #empty .hints { margin-top: 20px; display: flex; flex-direction: column; gap: 5px; align-items: center; }
  #empty .hint-chip {
    font-family: var(--mono); font-size: 10px;
    background: var(--surface); border: 1px solid var(--border);
    padding: 5px 12px; border-radius: 6px; color: var(--muted);
    cursor: pointer; transition: border-color 0.15s, color 0.15s;
  }
  #empty .hint-chip:hover { border-color: var(--accent); color: var(--accent); }


  /* ── Tabs ── */
  .tab-nav {
    display: flex;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    padding: 0 24px;
  }

  .tab-btn {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.06em;
    color: var(--muted);
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 12px 16px 10px;
    cursor: pointer;
    transition: color 0.15s, border-color 0.15s;
    margin-bottom: -1px;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

  .tab-panel { display: none; flex: 1; flex-direction: column; overflow: hidden; }
  .tab-panel.active { display: flex; }

  /* ── Insights panel ── */
  #insights-panel {
    flex: 1;
    overflow-y: auto;
    padding: 24px 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .insights-empty {
    text-align: center;
    padding: 60px 32px;
    color: var(--muted);
  }
  .insights-empty h2 { font-family: var(--mono); font-size: 15px; color: var(--text); margin-bottom: 8px; }
  .insights-empty p  { font-size: 12px; line-height: 1.6; }

  .insight-run-btn {
    margin: 0 auto;
    margin-top: 20px;
    display: block;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: var(--sans);
    font-size: 13px;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .insight-run-btn:hover { opacity: 0.85; }
  .insight-run-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .insight-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
  }

  .insight-section h3 {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
  }

  .insight-list { list-style: none; display: flex; flex-direction: column; gap: 8px; }

  .insight-list li {
    font-size: 12px;
    line-height: 1.5;
    color: var(--text);
    padding-left: 16px;
    position: relative;
  }
  .insight-list li::before { content: '·'; position: absolute; left: 0; color: var(--accent); }

  .source-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
  }
  .source-row:last-child { border-bottom: none; }

  .source-title { font-size: 12px; color: var(--text); font-weight: 500; }
  .source-meta  { font-family: var(--mono); font-size: 10px; color: var(--muted); }

  .score-bar {
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    margin-top: 4px;
    transition: width 0.4s ease;
  }

  .medal { margin-right: 6px; }

  .insight-lm {
    font-size: 12px;
    line-height: 1.7;
    color: var(--muted);
    white-space: pre-wrap;
    border-top: 1px solid var(--border);
    padding-top: 14px;
    margin-top: 4px;
  }

  #insights-loading {
    display: none;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 12px;
    font-family: var(--mono);
    padding: 40px 0;
    justify-content: center;
  }

  /* Input bar */
  #input-bar {
    border-top: 1px solid var(--border);
    padding: 14px 24px;
    flex-shrink: 0;
  }
  #input-inner { display: flex; gap: 10px; align-items: flex-end; }

  #question {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 300;
    resize: none;
    outline: none;
    line-height: 1.5;
    min-height: 42px;
    max-height: 120px;
    transition: border-color 0.15s;
  }
  #question:focus { border-color: var(--accent); }
  #question::placeholder { color: var(--muted); }

  #ask-btn {
    background: var(--accent); border: none; border-radius: 10px;
    width: 42px; height: 42px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: opacity 0.15s;
  }
  #ask-btn:hover { opacity: 0.85; }
  #ask-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  #ask-btn svg { width: 16px; height: 16px; fill: white; }

  /* Loading dots */
  .loading { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12px; font-family: var(--mono); padding: 6px 0; }
  .dot { width: 5px; height: 5px; background: var(--accent); border-radius: 50%; animation: pulse 1.2s ease-in-out infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes pulse { 0%,80%,100%{opacity:.2;transform:scale(.8)} 40%{opacity:1;transform:scale(1)} }

  #clear-btn {
    font-family: var(--mono); font-size: 10px;
    background: transparent; border: 1px solid var(--border);
    color: var(--muted); padding: 3px 8px; border-radius: 5px;
    cursor: pointer; float: right; margin-top: 4px;
    transition: border-color 0.15s, color 0.15s; display: none;
  }
  #clear-btn:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>

<header>
  <div class="logo">llm<span>/</span>wiki</div>
  <div id="stats">
    <span><b id="page-count">–</b> pages</span>
    <span id="conflict-stat" onclick="toggleConflicts()" style="display:none"></span>
  </div>
</header>

<div class="layout">

  <!-- ── Left: Add Content ── -->
  <div class="panel-left">
    <div class="panel-title">Add Content</div>

    <!-- URL clipper -->
    <div class="add-section">
      <h3>Clip a URL</h3>
      <input class="url-input" id="url-input" type="url" placeholder="https://article.com/...">
      <button class="btn btn-primary" id="clip-btn" onclick="clipUrl()">Clip & Compile</button>
    </div>

    <!-- PDF upload -->
    <div class="add-section">
      <h3>Upload a PDF</h3>
      <div class="drop-zone" id="drop-zone">
        <input type="file" accept=".pdf" id="pdf-input" onchange="uploadPdf(this.files[0])">
        <div class="drop-icon">📄</div>
        <div class="drop-label"><b>Drop a PDF</b> or click to browse</div>
      </div>
    </div>

    <!-- Compile log -->
    <div class="log-area">
      <div class="log-title">Compile log</div>
      <div id="compile-log">Ready.</div>
    </div>

    <!-- Conflicts -->
    <div id="conflicts-panel">
      <div class="conflict-title">⚠️ Conflicts detected</div>
      <div id="conflicts-content"></div>
    </div>
  </div>

  <!-- ── Right: Tabs ── -->
  <div class="panel-right">

    <!-- Tab nav -->
    <div class="tab-nav">
      <button class="tab-btn active" onclick="switchTab('chat')">Chat</button>
      <button class="tab-btn" onclick="switchTab('insights')">Insights</button>
    </div>

    <!-- Chat tab -->
    <div class="tab-panel active" id="tab-chat">
      <div id="history">
        <div id="empty">
          <h2>Your personal knowledge base</h2>
          <p>Add content on the left. Ask questions here.</p>
          <div class="hints">
            <div class="hint-chip" onclick="useHint(this)">What is self-attention?</div>
            <div class="hint-chip" onclick="useHint(this)">How does GPT differ from BERT?</div>
            <div class="hint-chip" onclick="useHint(this)">Summarise what I know about transformers</div>
          </div>
        </div>
      </div>

      <div id="input-bar">
        <button id="clear-btn" onclick="clearHistory()">clear chat</button>
        <div id="input-inner">
          <textarea id="question" placeholder="Ask your wiki anything..." rows="1"></textarea>
          <button id="ask-btn" onclick="ask()">
            <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Insights tab -->
    <div class="tab-panel" id="tab-insights">
      <div id="insights-panel">
        <div class="insights-empty" id="insights-empty">
          <h2>Wiki Insights</h2>
          <p>Detect knowledge gaps and see which sources contributed most.</p>
          <button class="insight-run-btn" id="insights-run-btn" onclick="runInsights()">Run Analysis</button>
        </div>
        <div id="insights-loading">
          <div class="dot"></div><div class="dot"></div><div class="dot"></div>
          <span>Analysing your wiki... this takes ~30s</span>
        </div>
        <div id="insights-results" style="display:none; flex-direction:column; gap:20px;"></div>
      </div>
    </div>

  </div>

</div>

<script>
  let history = [];

  // ── Auto-resize textarea ──
  const textarea = document.getElementById('question');
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
  });
  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
  });

  // ── Stats ──
  async function loadStats() {
    const res  = await fetch('/stats');
    const data = await res.json();
    document.getElementById('page-count').textContent = data.page_count;
    if (data.conflict_count > 0) {
      const el = document.getElementById('conflict-stat');
      el.textContent = `⚠️ ${data.conflict_count} conflict(s)`;
      el.style.display = 'inline';
    }
  }

  // ── Conflicts panel ──
  async function toggleConflicts() {
    const panel = document.getElementById('conflicts-panel');
    if (panel.style.display === 'block') {
      panel.style.display = 'none';
      return;
    }
    const res  = await fetch('/conflicts');
    const data = await res.json();
    document.getElementById('conflicts-content').textContent = data.content || 'No conflicts logged.';
    panel.style.display = 'block';
  }

  // ── Compile log ──
  function logLine(text) {
    const log = document.getElementById('compile-log');
    const line = document.createElement('div');
    if (text.includes('✅')) line.className = 'log-done';
    else if (text.includes('⚠️')) line.className = 'log-warn';
    else if (text.includes('❌')) line.className = 'log-err';
    line.textContent = text;
    if (log.textContent === 'Ready.') log.textContent = '';
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  function clearLog() {
    document.getElementById('compile-log').innerHTML = '';
  }

  function setInputsDisabled(disabled) {
    document.getElementById('clip-btn').disabled = disabled;
    document.getElementById('pdf-input').disabled = disabled;
  }

  async function runCompileStream(filename) {
    return new Promise((resolve) => {
      const es = new EventSource(`/compile-stream?file=${encodeURIComponent(filename)}`);
      es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.log)    logLine(data.log);
        if (data.status) { es.close(); resolve(data.status); }
      };
      es.onerror = () => { es.close(); resolve('error'); };
    });
  }

  // ── URL Clipper ──
  async function clipUrl() {
    const url = document.getElementById('url-input').value.trim();
    if (!url) return;
    clearLog();
    setInputsDisabled(true);
    logLine(`📎 Clipping: ${url}`);

    try {
      const res  = await fetch('/ingest-url', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url})
      });
      const data = await res.json();
      if (data.error) { logLine(`❌ ${data.error}`); setInputsDisabled(false); return; }
      logLine(`💾 ${data.message}`);
      logLine('🚀 Compiling...');
      const status = await runCompileStream(data.filename);
      if (status === 'done') {
        logLine('🎉 Done!');
        loadStats();
        document.getElementById('url-input').value = '';
      } else {
        logLine('❌ Compile failed. Check terminal for details.');
      }
    } catch(e) {
      logLine(`❌ ${e.message}`);
    }
    setInputsDisabled(false);
  }

  // ── PDF Upload ──
  async function uploadPdf(file) {
    if (!file) return;
    clearLog();
    setInputsDisabled(true);
    logLine(`📄 Ingesting: ${file.name}`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res  = await fetch('/ingest-pdf', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) { logLine(`❌ ${data.error}`); setInputsDisabled(false); return; }
      logLine(`💾 ${data.message}`);
      logLine('🚀 Compiling...');
      const status = await runCompileStream(data.filename);
      if (status === 'done') {
        logLine('🎉 Done!');
        loadStats();
      } else {
        logLine('❌ Compile failed. Check terminal for details.');
      }
    } catch(e) {
      logLine(`❌ ${e.message}`);
    }
    setInputsDisabled(false);
    document.getElementById('pdf-input').value = '';
  }

  // Drag-over styling
  const dz = document.getElementById('drop-zone');
  dz.addEventListener('dragover', () => dz.classList.add('drag-over'));
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', () => dz.classList.remove('drag-over'));

  // ── Chat ──
  function useHint(el) {
    document.getElementById('question').value = el.textContent;
    ask();
  }

  async function ask() {
    const q = textarea.value.trim();
    if (!q) return;

    document.getElementById('empty').style.display  = 'none';
    document.getElementById('clear-btn').style.display = 'block';
    textarea.value = '';
    textarea.style.height = 'auto';
    document.getElementById('ask-btn').disabled = true;

    const turn = document.createElement('div');
    turn.className = 'turn';
    turn.innerHTML = `<div class="question">${escapeHtml(q)}</div>`;

    const loading = document.createElement('div');
    loading.className = 'loading';
    loading.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div><span>thinking...</span>';
    turn.appendChild(loading);
    document.getElementById('history').appendChild(turn);
    scrollBottom();

    try {
      const res  = await fetch('/query', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q, history})
      });
      const data = await res.json();
      loading.remove();

      const wrap = document.createElement('div');
      wrap.className = 'answer-wrap';
      const ans = document.createElement('div');
      ans.className = 'answer';
      ans.textContent = data.answer;
      wrap.appendChild(ans);

      if (data.citations && data.citations.length) {
        const cits = document.createElement('div');
        cits.className = 'citations';
        data.citations.forEach(c => {
          const tag = document.createElement('span');
          tag.className = 'citation-tag';
          tag.textContent = c;
          cits.appendChild(tag);
        });
        wrap.appendChild(cits);
      }
      turn.appendChild(wrap);
      history.push({question: q, answer: data.answer});
    } catch(e) {
      loading.remove();
      const err = document.createElement('div');
      err.className = 'answer';
      err.style.color = 'var(--red)';
      err.textContent = 'Something went wrong. Check your terminal.';
      turn.appendChild(err);
    }

    document.getElementById('ask-btn').disabled = false;
    scrollBottom();
  }

  function clearHistory() {
    history = [];
    const h = document.getElementById('history');
    h.innerHTML = `<div id="empty">
      <h2>Your personal knowledge base</h2>
      <p>Add content on the left. Ask questions here.</p>
      <div class="hints">
        <div class="hint-chip" onclick="useHint(this)">What is self-attention?</div>
        <div class="hint-chip" onclick="useHint(this)">How does GPT differ from BERT?</div>
        <div class="hint-chip" onclick="useHint(this)">Summarise what I know about transformers</div>
      </div>
    </div>`;
    document.getElementById('clear-btn').style.display = 'none';
  }

  function scrollBottom() { document.getElementById('history').scrollTop = 9999999; }

  function escapeHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }


  // ── Tab switching ──
  function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => {
      b.classList.toggle('active', (i === 0 && tab === 'chat') || (i === 1 && tab === 'insights'));
    });
    document.getElementById('tab-chat').classList.toggle('active', tab === 'chat');
    document.getElementById('tab-insights').classList.toggle('active', tab === 'insights');
  }

  // ── Insights ──
  async function runInsights() {
    document.getElementById('insights-empty').style.display    = 'none';
    document.getElementById('insights-loading').style.display  = 'flex';
    document.getElementById('insights-results').style.display  = 'none';
    document.getElementById('insights-run-btn').disabled = true;

    try {
      const res  = await fetch('/insights');
      const data = await res.json();

      document.getElementById('insights-loading').style.display = 'none';

      if (data.error) {
        document.getElementById('insights-empty').style.display = 'block';
        document.getElementById('insights-empty').querySelector('p').textContent = data.error;
        document.getElementById('insights-run-btn').disabled = false;
        return;
      }

      const results = document.getElementById('insights-results');
      results.innerHTML = '';

      // ── Gap section ──
      if (data.gaps && !data.gaps.error) {
        const g = data.gaps;

        if (g.missing_topics && g.missing_topics.length) {
          results.appendChild(insightSection('❓ Missing Topics',
            g.missing_topics.map(t => `<li>${escapeHtml(t)}</li>`).join('')
          ));
        }

        if (g.orphan_links && g.orphan_links.length) {
          results.appendChild(insightSection('🔗 Orphan Links — Referenced But No Page',
            g.orphan_links.map(l => `<li>${escapeHtml(l)}</li>`).join('')
          ));
        }

        if (g.thin_pages && Object.keys(g.thin_pages).length) {
          const items = Object.entries(g.thin_pages)
            .sort((a,b) => a[1]-b[1])
            .map(([stem, wc]) => `<li>${escapeHtml(stem)} <span style="color:var(--muted)">(${wc} words)</span></li>`)
            .join('');
          results.appendChild(insightSection('📄 Thin Pages — Need More Content', items));
        }

        if (g.suggested_reading && g.suggested_reading.length) {
          results.appendChild(insightSection('📚 Suggested Reading',
            g.suggested_reading.map(t => `<li>${escapeHtml(t)}</li>`).join('')
          ));
        }
      }

      // ── Ranking section ──
      if (data.ranking && !data.ranking.error) {
        const r = data.ranking;
        const maxScore = Math.max(...r.ranked_sources.map(s => s.score), 1);

        const rows = r.ranked_sources.map((s, i) => {
          const medals = ['🥇','🥈','🥉'];
          const medal  = i < 3 ? medals[i] : `${i+1}.`;
          const pct    = Math.round((s.score / maxScore) * 100);
          return `<div class="source-row">
            <div class="source-title"><span class="medal">${medal}</span>${escapeHtml(s.title)}</div>
            <div class="source-meta">Score: ${s.score} · Cited by ${s.citation_count} pages · ${s.word_count.toLocaleString()} words</div>
            <div class="score-bar" style="width:${pct}%"></div>
          </div>`;
        }).join('');

        const sec = insightSection('📊 Source Rankings', rows, false);
        if (r.insight) {
          const lm = document.createElement('div');
          lm.className = 'insight-lm';
          lm.textContent = r.insight;
          sec.appendChild(lm);
        }
        results.appendChild(sec);
      }

      results.style.display = 'flex';

      // Re-run button at bottom
      const rerun = document.createElement('button');
      rerun.className = 'insight-run-btn';
      rerun.textContent = 'Re-run Analysis';
      rerun.onclick = runInsights;
      results.appendChild(rerun);

    } catch(e) {
      document.getElementById('insights-loading').style.display = 'none';
      document.getElementById('insights-empty').style.display   = 'block';
      document.getElementById('insights-empty').querySelector('p').textContent = 'Something went wrong. Check your terminal.';
      document.getElementById('insights-run-btn').disabled = false;
    }
  }

  function insightSection(title, itemsHtml, isList = true) {
    const sec = document.createElement('div');
    sec.className = 'insight-section';
    sec.innerHTML = `<h3>${title}</h3>${isList ? '<ul class="insight-list">' + itemsHtml + '</ul>' : itemsHtml}`;
    return sec;
  }

  loadStats();
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/stats")
def stats():
    return jsonify(get_wiki_stats())

@app.route("/conflicts")
def conflicts():
    return jsonify({"content": get_conflicts()})

@app.route("/query", methods=["POST"])
def query():
    data     = request.get_json()
    question = data.get("question", "").strip()
    history  = data.get("history", [])
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        return jsonify(query_wiki(question, history))
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ingest-url", methods=["POST"])
def ingest_url_route():
    data = request.get_json()
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    filename, message = ingest_url(url)
    if not filename:
        return jsonify({"error": message}), 400
    return jsonify({"filename": filename, "message": message})

@app.route("/ingest-pdf", methods=["POST"])
def ingest_pdf_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    filename, message = ingest_pdf(f.read(), f.filename)
    if not filename:
        return jsonify({"error": message}), 400
    return jsonify({"filename": filename, "message": message})

@app.route("/compile-stream")
def compile_stream():
    filename = request.args.get("file", "")
    if not filename:
        return jsonify({"error": "No file specified"}), 400
    return Response(
        stream_with_context(run_compile_stream(filename)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/insights")
def insights_route():
    """Run gap detection and source ranking directly — no subprocess."""
    try:
        # ── Import insights functions directly ──
        import re as _re
        from pathlib import Path as _Path

        wiki_dir    = BASE_DIR / "wiki"
        sources_dir = BASE_DIR / "sources"
        index_file  = wiki_dir / "INDEX.md"
        compiled_log= BASE_DIR / ".compiled"

        def _load_index():
            return index_file.read_text(encoding="utf-8") if index_file.exists() else ""

        def _load_pages():
            pages = {}
            for f in wiki_dir.glob("*.md"):
                if f.name in ("INDEX.md", "CONFLICTS.md", "_raw_output.md"):
                    continue
                pages[f.stem] = f.read_text(encoding="utf-8", errors="replace")
            return pages

        def _load_sources():
            compiled = set()
            if compiled_log.exists():
                compiled = set(compiled_log.read_text().splitlines())
            sources = {}
            for f in sources_dir.glob("*"):
                if f.is_file() and f.suffix in {".txt", ".md"} and f.name in compiled:
                    sources[f.name] = f.read_text(encoding="utf-8", errors="replace")
            return sources

        def _call(messages, max_tokens=1500):
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=0.3,
                        top_p=0.95,
                        max_tokens=max_tokens,
                        extra_body={"chat_template_kwargs": {"thinking": False}},
                        stream=False,
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    if attempt < 2:
                        time.sleep(10)
                    else:
                        raise

        pages      = _load_pages()
        sources    = _load_sources()
        wiki_index = _load_index()

        # ── Gap detection ──
        gaps = {"error": "Wiki is empty — compile some sources first."}
        if pages:
            all_links = set()
            for content in pages.values():
                all_links.update(_re.findall(r"\[\[([^\]]+)\]\]", content))

            existing     = set(pages.keys())
            orphan_links = sorted(all_links - existing - {"INDEX"})
            page_lengths = {stem: len(c.split()) for stem, c in pages.items()}
            thin_pages   = {s: wc for s, wc in page_lengths.items() if wc < 120}
            page_summary = "\n".join(
                f"- {s} ({wc} words)" for s, wc in sorted(page_lengths.items(), key=lambda x: -x[1])
            )

            raw = _call([
                {"role": "system", "content": (
                    "You are a knowledge curator analysing a personal wiki. "
                    "Based on the wiki's current pages and index, identify:\n"
                    "1. Important related topics completely missing\n"
                    "2. Existing topics that need much more depth\n"
                    "3. Specific sources the person should read next\n\n"
                    "Format your response as:\n\n"
                    "MISSING_TOPICS\n<bullet list>\nEND\n\n"
                    "NEEDS_DEPTH\n<bullet list>\nEND\n\n"
                    "SUGGESTED_READING\n<bullet list>\nEND"
                )},
                {"role": "user", "content": (
                    f"## Wiki Index\n{wiki_index}\n\n"
                    f"## All Pages\n{page_summary}\n\n"
                    f"## Orphan Links\n" +
                    ("\n".join(f"- {l}" for l in orphan_links) if orphan_links else "None")
                )},
            ])

            def _section(text, tag):
                m = _re.search(rf"{tag}\n(.*?)END", text, re.DOTALL)
                if not m:
                    return []
                return [
                    l.strip().lstrip("-•* ").strip()
                    for l in m.group(1).strip().splitlines()
                    if l.strip() and l.strip() not in ("-", "•", "*")
                ]

            gaps = {
                "orphan_links":     orphan_links,
                "thin_pages":       thin_pages,
                "missing_topics":   _section(raw, "MISSING_TOPICS"),
                "needs_depth":      _section(raw, "NEEDS_DEPTH"),
                "suggested_reading":_section(raw, "SUGGESTED_READING"),
                "page_count":       len(pages),
                "total_words":      sum(page_lengths.values()),
            }

        # ── Source ranking ──
        ranking = {"error": "No compiled sources found."}
        if sources and pages:
            scores = []
            for name, src_content in sources.items():
                citation_count = sum(
                    1 for c in pages.values()
                    if name in c or _Path(name).stem in c
                )
                links_in_src = len(_re.findall(r"\[\[([^\]]+)\]\]", src_content))
                wc = len(src_content.split())
                score = (citation_count * 10) + (links_in_src * 2) + min(wc // 100, 5)

                title_m = _re.search(r"^#\s+(.+)$", src_content, re.MULTILINE)
                title   = title_m.group(1).strip()[:60] if title_m else _Path(name).stem
                url_m   = _re.search(r"\*\*Source URL:\*\*\s*(.+)", src_content)
                url     = url_m.group(1).strip() if url_m else None

                scores.append({
                    "filename": name, "title": title, "url": url,
                    "citation_count": citation_count,
                    "word_count": wc, "score": score,
                })

            scores.sort(key=lambda x: -x["score"])

            top = "\n".join(
                f"{i+1}. {s['title']} (score:{s['score']}, cited by {s['citation_count']} pages)"
                for i, s in enumerate(scores[:10])
            )
            low = "\n".join(
                f"- {s['title']}" for s in scores if s["score"] == 0
            )

            insight = _call([
                {"role": "system", "content": (
                    "You are a knowledge curator. Given ranked sources and a wiki index, "
                    "give brief qualitative insight in 3 short paragraphs:\n"
                    "1. Why the top sources contributed most\n"
                    "2. Why low-scoring sources added little\n"
                    "3. One recommendation going forward"
                )},
                {"role": "user", "content": (
                    f"## Wiki Index\n{wiki_index}\n\n"
                    f"## Top Sources\n{top}\n\n"
                    f"## Low/Zero Sources\n{low if low else 'None'}"
                )},
            ], max_tokens=600)

            ranking = {
                "ranked_sources": scores,
                "insight":        insight,
                "total_sources":  len(scores),
            }

        return jsonify({"gaps": gaps, "ranking": ranking})

    except Exception as e:
        print(f"Insights error: {e}")
        return jsonify({"error": str(e)}), 500

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = get_wiki_stats()
    print(f"\n🌐  LLM Wiki UI")
    print(f"    {s['page_count']} wiki pages loaded")
    print(f"    Open: http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)
