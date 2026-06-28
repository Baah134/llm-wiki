"""
compiler.py — LLM Wiki Compiler Agent
Uses NVIDIA NIM to compile raw sources into a structured wiki.

Usage:
    python compiler.py                  # compile all new sources
    python compiler.py --file doc.txt   # compile a specific file
    python compiler.py --query "What is X?"  # ask a question against the wiki
    python compiler.py --all            # recompile everything from scratch
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent
SOURCES_DIR  = BASE_DIR / "sources"
WIKI_DIR     = BASE_DIR / "wiki"
AGENT_MD     = BASE_DIR / "AGENT.md"
INDEX_FILE   = WIKI_DIR / "INDEX.md"
COMPILED_LOG = BASE_DIR / ".compiled"

MODEL      = "meta/llama-3.3-70b-instruct"
MAX_TOKENS = 8192

# ── NVIDIA NIM client ─────────────────────────────────────────────────────────

def get_client():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("❌  NVIDIA_API_KEY not set.")
        print("    Export it first:  export NVIDIA_API_KEY=nvapi-xxxx")
        sys.exit(1)
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_agent_schema() -> str:
    return AGENT_MD.read_text(encoding="utf-8")

def load_wiki_index() -> str:
    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8")
    return "_(Wiki is empty — no index yet.)_"

def load_wiki_page(name: str) -> str:
    """Load a single wiki page by name (with or without .md)."""
    if not name.endswith(".md"):
        name = name + ".md"
    path = WIKI_DIR / name
    if path.exists():
        return f"### FILE: {name}\n\n{path.read_text(encoding='utf-8')}"
    return ""

def load_all_wiki_pages() -> str:
    """Load every wiki page — used for queries."""
    pages = []
    for f in sorted(WIKI_DIR.glob("*.md")):
        if f.name == "INDEX.md":
            continue
        pages.append(f"### FILE: {f.name}\n\n{f.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(pages) if pages else "_(No wiki pages exist yet.)_"

def get_compiled_files() -> set:
    if not Path(COMPILED_LOG).exists():
        return set()
    return set(Path(COMPILED_LOG).read_text().splitlines())

def mark_compiled(filename: str):
    with open(COMPILED_LOG, "a") as f:
        f.write(filename + "\n")

def call_api(client: OpenAI, messages: list, max_tokens: int = MAX_TOKENS, label: str = "") -> str:
    """Call the API with automatic retry on timeout/server errors."""
    for attempt in range(3):
        try:
            if label:
                print(f"   🤖  {label}...")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                top_p=0.95,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                print(f"   ⚠️  Attempt {attempt + 1} failed ({e}). Retrying in 10s...")
                time.sleep(10)
            else:
                raise

def save_wiki_output(response_text: str) -> list:
    """Parse FILE blocks from agent response and write them to disk."""
    pattern = r"---\s*FILE:\s*(wiki/[^\s]+\.md)\s*---\n(.*?)---\s*END FILE\s*---"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if not matches:
        print("\n⚠️  Agent response didn't use expected FILE blocks.")
        print("   Raw output saved to wiki/_raw_output.md")
        (WIKI_DIR / "_raw_output.md").write_text(response_text, encoding="utf-8")
        return []

    saved = []
    for filepath, content in matches:
        full_path = BASE_DIR / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.strip(), encoding="utf-8")
        saved.append(filepath)
        print(f"   ✅  Saved {filepath}")
    return saved

# ── Step 1: Index-first — ask which pages are relevant ───────────────────────

def get_relevant_pages(client: OpenAI, source_name: str, raw_content: str) -> list[str]:
    """
    Send only the index + source to the model.
    Ask it which existing wiki pages are relevant.
    Returns a list of page filenames.
    """
    wiki_index = load_wiki_index()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a wiki librarian. Given a new source document and a wiki index, "
                "identify which existing wiki pages are relevant to the new source. "
                "Reply with ONLY a comma-separated list of page filenames (e.g. Transformer.md, Attention.md). "
                "If no pages are relevant, reply with: none"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Wiki Index\n{wiki_index}\n\n"
                f"## New Source: {source_name}\n{raw_content[:3000]}"
                # We only send the first 3000 chars for the relevance check — fast + cheap
            ),
        },
    ]

    result = call_api(client, messages, max_tokens=256, label="Checking index for relevant pages")
    result = result.strip()

    if result.lower() == "none" or not result:
        return []

    # Parse the comma-separated list
    pages = [p.strip().replace("[[", "").replace("]]", "") for p in result.split(",")]
    pages = [p if p.endswith(".md") else p + ".md" for p in pages if p]
    return pages

# ── Step 2: Full compilation with only relevant pages ────────────────────────

def compile_with_context(client: OpenAI, source_path: Path, relevant_pages: list[str]) -> list[str]:
    """Run the full compilation using only the relevant wiki pages as context."""
    raw_content  = source_path.read_text(encoding="utf-8", errors="replace")
    agent_schema = load_agent_schema()
    wiki_index   = load_wiki_index()
    today        = datetime.now().strftime("%Y-%m-%d")

    # Load only the relevant pages instead of everything
    if relevant_pages:
        page_context = "\n\n---\n\n".join(
            load_wiki_page(p) for p in relevant_pages if load_wiki_page(p)
        )
        pages_note = f"Relevant existing pages loaded: {', '.join(relevant_pages)}"
    else:
        page_context = "_(No existing pages are relevant to this source.)_"
        pages_note   = "No existing pages loaded — this source introduces new concepts."

    system_prompt = f"""
{agent_schema}

Today's date: {today}

## Current Wiki Index
{wiki_index}

## Relevant Existing Wiki Pages
{page_context}

## Note
{pages_note}

## Output Format (REQUIRED)
For every wiki page you create or update, wrap it exactly like this:

--- FILE: wiki/PageName.md ---
<full markdown content of the page>
--- END FILE ---

You may output multiple FILE blocks in one response.
Always include an updated INDEX.md as one of the FILE blocks.
""".strip()

    user_prompt = f"""
New source to compile into the wiki:

**Filename:** {source_path.name}

**Content:**
{raw_content}

Read this source carefully. Then:
1. Create or update wiki pages for every significant concept, person, method, or tool mentioned.
2. Cross-link pages using [[wiki-links]].
3. Update wiki/INDEX.md to reflect any new or changed pages.

Output all changed files using the FILE block format.
""".strip()

    response_text = call_api(
        client,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        label="Compiling source into wiki",
    )
    return save_wiki_output(response_text)

# ── Step 3: Agent loop — check own output for missing pages ──────────────────

def fill_missing_stubs(client: OpenAI, saved_pages: list[str]):
    """
    Re-read the pages just written.
    Find [[wiki-links]] that point to pages that don't exist yet.
    Create stub pages for them automatically.
    """
    # Collect all [[links]] from newly saved pages
    all_links = set()
    for filepath in saved_pages:
        content = (BASE_DIR / filepath).read_text(encoding="utf-8")
        found = re.findall(r"\[\[([^\]]+)\]\]", content)
        all_links.update(found)

    # Find which ones don't have a page yet
    existing = {f.stem for f in WIKI_DIR.glob("*.md")}
    missing  = [link for link in all_links if link not in existing and link != "INDEX"]

    if not missing:
        print("   ✅  No missing linked pages — wiki is consistent.")
        return

    print(f"   🔍  Found {len(missing)} unresolved link(s): {', '.join(missing)}")
    print("   🤖  Creating stubs for missing pages...")

    wiki_index = load_wiki_index()

    system_prompt = """
You are a wiki maintainer. Create brief stub pages for concepts that are referenced
but don't have their own page yet. Each stub should have:
- A one-paragraph summary of what the concept is
- A "Related" field with wiki-links to connected concepts
- A "Open Questions" section noting what needs to be filled in later

Use the standard wiki page format with FILE blocks.
""".strip()

    user_prompt = f"""
## Wiki Index (for context)
{wiki_index}

## Missing pages to create stubs for:
{chr(10).join(f'- {m}' for m in missing)}

Create a stub wiki page for each missing concept.
Output each as a FILE block:

--- FILE: wiki/PageName.md ---
<stub content>
--- END FILE ---
""".strip()

    response_text = call_api(
        client,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=4096,
        label="Creating stubs for missing pages",
    )
    stub_saved = save_wiki_output(response_text)
    if stub_saved:
        print(f"   ✅  Created {len(stub_saved)} stub page(s).")

# ── Step 4: Conflict detector ─────────────────────────────────────────────────

def detect_conflicts(client: OpenAI, source_path: Path, relevant_pages: list[str]) -> list[dict]:
    """
    Compare new source against existing relevant wiki pages.
    Returns a list of conflicts: [{page, claim, conflict}]
    """
    if not relevant_pages:
        return []

    raw_content = source_path.read_text(encoding="utf-8", errors="replace")
    page_context = "\n\n---\n\n".join(
        load_wiki_page(p) for p in relevant_pages if load_wiki_page(p)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fact-checker for a personal wiki. "
                "Compare a new source document against existing wiki pages. "
                "Find any direct contradictions — claims in the new source that conflict with "
                "established facts in the wiki. Ignore differences in emphasis or detail; "
                "only flag genuine factual contradictions. "
                "Reply in this exact format for each conflict found:\n\n"
                "CONFLICT\n"
                "Page: <wiki page name>\n"
                "Wiki says: <what the wiki currently claims>\n"
                "Source says: <what the new source claims>\n"
                "END\n\n"
                "If no conflicts are found, reply with exactly: NO CONFLICTS"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Existing Wiki Pages\n{page_context}\n\n"
                f"## New Source: {source_path.name}\n{raw_content[:4000]}"
            ),
        },
    ]

    result = call_api(client, messages, max_tokens=1024, label="Checking for conflicts")

    if "NO CONFLICTS" in result:
        return []

    # Parse conflict blocks
    conflicts = []
    blocks = re.findall(r"CONFLICT\n(.*?)END", result, re.DOTALL)
    for block in blocks:
        page    = re.search(r"Page:\s*(.+)", block)
        wiki    = re.search(r"Wiki says:\s*(.+)", block)
        source  = re.search(r"Source says:\s*(.+)", block)
        if page and wiki and source:
            conflicts.append({
                "page":    page.group(1).strip(),
                "wiki":    wiki.group(1).strip(),
                "source":  source.group(1).strip(),
            })
    return conflicts

def save_conflicts(source_name: str, conflicts: list[dict]):
    """Append conflicts to a CONFLICTS.md log in the wiki folder."""
    if not conflicts:
        return
    log_path = WIKI_DIR / "CONFLICTS.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n## {source_name} — {timestamp}\n"]
    for c in conflicts:
        lines.append(f"- **{c['page']}**")
        lines.append(f"  - Wiki says: {c['wiki']}")
        lines.append(f"  - Source says: {c['source']}\n")
    with open(log_path, "a") as f:
        f.write("\n".join(lines))

# ── Orchestrator ──────────────────────────────────────────────────────────────

def compile_source(client: OpenAI, source_path: Path) -> list[dict]:
    """
    Full compile pipeline. Returns list of conflicts found (empty if none).
    """
    print(f"\n📄  Compiling: {source_path.name}")

    raw_content = source_path.read_text(encoding="utf-8", errors="replace")

    # Step 1 — index-first: find relevant pages without loading everything
    relevant_pages = get_relevant_pages(client, source_path.name, raw_content)
    if relevant_pages:
        print(f"   📎  Relevant pages: {', '.join(relevant_pages)}")
    else:
        print("   📎  No existing pages relevant — starting fresh.")

    # Step 2 — conflict detection before writing anything
    conflicts = detect_conflicts(client, source_path, relevant_pages)
    if conflicts:
        print(f"\n   ⚠️   {len(conflicts)} conflict(s) found in new source:")
        for c in conflicts:
            print(f"        [{c['page']}] Wiki: {c['wiki']}")
            print(f"               Source: {c['source']}")
        save_conflicts(source_path.name, conflicts)
        print(f"   📝  Conflicts logged to wiki/CONFLICTS.md")
    else:
        print("   ✅  No conflicts detected.")

    # Step 3 — compile with only the relevant context
    saved = compile_with_context(client, source_path, relevant_pages)

    # Step 4 — agent loop: check own output and fill missing stubs
    if saved:
        fill_missing_stubs(client, saved)

    mark_compiled(source_path.name)
    print(f"\n   ✅  Done. {len(saved)} wiki file(s) written.")
    return conflicts

# ── Core: Query the wiki ──────────────────────────────────────────────────────

def query_wiki(client: OpenAI, question: str):
    print(f"\n🔍  Query: {question}\n")

    wiki_index    = load_wiki_index()
    existing_wiki = load_all_wiki_pages()

    system_prompt = f"""
You are a knowledgeable assistant with access to a personal wiki.
Answer questions using ONLY the information in the wiki pages below.
If the answer isn't in the wiki, say so clearly.
Cite which wiki page(s) your answer comes from.

## Wiki Index
{wiki_index}

## Wiki Pages
{existing_wiki}
""".strip()

    response = call_api(
        client,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        max_tokens=4096,
        label="Querying wiki",
    )
    print(response)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Wiki Compiler Agent")
    parser.add_argument("--file",  type=str, help="Compile a specific file from sources/")
    parser.add_argument("--query", type=str, help="Ask a question against the wiki")
    parser.add_argument("--all",   action="store_true", help="Recompile all sources")
    args = parser.parse_args()

    WIKI_DIR.mkdir(exist_ok=True)
    client = get_client()

    if args.query:
        query_wiki(client, args.query)
        return

    if args.file:
        path = SOURCES_DIR / args.file
        if not path.exists():
            print(f"❌  File not found: {path}")
            sys.exit(1)
        compile_source(client, path)
        return

    # Default: compile all new sources
    compiled = get_compiled_files() if not args.all else set()
    source_files = [
        f for f in sorted(SOURCES_DIR.iterdir())
        if f.is_file() and f.suffix in {".txt", ".md", ".pdf"} and f.name not in compiled
    ]

    if not source_files:
        print("✅  No new sources to compile. Drop files into sources/ and run again.")
        print("    Use --all to recompile everything from scratch.")
        return

    print(f"📚  Found {len(source_files)} new source(s) to compile.\n")
    for f in source_files:
        compile_source(client, f)

    print(f"\n🎉  Wiki compilation complete!")
    print(f"    Wiki pages are in: {WIKI_DIR}/")

if __name__ == "__main__":
    main()
