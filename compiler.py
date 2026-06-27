"""
compiler.py — LLM Wiki Compiler Agent
Uses NVIDIA NIM (DeepSeek) to compile raw sources into a structured wiki.

Usage:
    python compiler.py                  # compile all new sources
    python compiler.py --file doc.txt   # compile a specific file
    python compiler.py --query "What is X?"  # ask a question against the wiki
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
SOURCES_DIR = BASE_DIR / "sources"
WIKI_DIR    = BASE_DIR / "wiki"
AGENT_MD    = BASE_DIR / "AGENT.md"
INDEX_FILE  = WIKI_DIR / "INDEX.md"
COMPILED_LOG= BASE_DIR / ".compiled"   # tracks which files have been compiled

MODEL       = "meta/llama-3.3-70b-instruct"
MAX_TOKENS  = 8192

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

def load_wiki_pages() -> str:
    """Load all existing wiki pages into one string for context."""
    pages = []
    for f in sorted(WIKI_DIR.glob("*.md")):
        if f.name == "INDEX.md":
            continue
        pages.append(f"### FILE: {f.name}\n\n{f.read_text(encoding='utf-8')}")
    if not pages:
        return "_(No wiki pages exist yet.)_"
    return "\n\n---\n\n".join(pages)

def get_compiled_files() -> set:
    if not Path(COMPILED_LOG).exists():
        return set()
    return set(Path(COMPILED_LOG).read_text().splitlines())

def mark_compiled(filename: str):
    with open(COMPILED_LOG, "a") as f:
        f.write(filename + "\n")

def save_wiki_output(response_text: str):
    """
    Parse the agent's response and save each wiki page it returns.
    The agent is prompted to wrap each file in:
        --- FILE: wiki/SomePage.md ---
        <content>
        --- END FILE ---
    """
    import re
    pattern = r"---\s*FILE:\s*(wiki/[^\s]+\.md)\s*---\n(.*?)---\s*END FILE\s*---"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if not matches:
        # Fallback: if agent didn't use the format, print raw output
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

# ── Core: Compile a source file ───────────────────────────────────────────────

def compile_source(client: OpenAI, source_path: Path):
    print(f"\n📄  Compiling: {source_path.name}")

    raw_content = source_path.read_text(encoding="utf-8", errors="replace")
    agent_schema = load_agent_schema()
    wiki_index   = load_wiki_index()
    existing_wiki = load_wiki_pages()
    today = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
{agent_schema}

Today's date: {today}

## Current Wiki Index
{wiki_index}

## Existing Wiki Pages
{existing_wiki}

## Output Format (REQUIRED)
For every wiki page you create or update, wrap it exactly like this:

--- FILE: wiki/PageName.md ---
<full markdown content of the page>
--- END FILE ---

You may output multiple FILE blocks in one response.
Always include an updated INDEX file as one of the FILE blocks.
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

    print("   🤖  Calling DeepSeek via NVIDIA NIM...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,      # lower = more consistent/structured output
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"thinking": False}},
        stream=False,
    )

    response_text = response.choices[0].message.content
    saved = save_wiki_output(response_text)
    mark_compiled(source_path.name)
    print(f"   ✅  Done. {len(saved)} wiki file(s) written.")

# ── Core: Query the wiki ──────────────────────────────────────────────────────

def query_wiki(client: OpenAI, question: str):
    print(f"\n🔍  Query: {question}\n")

    agent_schema  = load_agent_schema()
    wiki_index    = load_wiki_index()
    existing_wiki = load_wiki_pages()

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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.4,
        top_p=0.95,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"thinking": False}},
        stream=False,
    )

    print(response.choices[0].message.content)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Wiki Compiler")
    parser.add_argument("--file",  type=str, help="Compile a specific file from sources/")
    parser.add_argument("--query", type=str, help="Ask a question against the wiki")
    parser.add_argument("--all",   action="store_true", help="Recompile all sources (including already compiled)")
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

    # Default: compile all new (uncompiled) sources
    compiled = get_compiled_files() if not args.all else set()
    source_files = [
        f for f in sorted(SOURCES_DIR.iterdir())
        if f.is_file() and f.suffix in {".txt", ".md", ".pdf"} and f.name not in compiled
    ]

    if not source_files:
        print("✅  No new sources to compile. Drop files into sources/ and run again.")
        print("    Use --all to recompile everything.")
        return

    print(f"📚  Found {len(source_files)} new source(s) to compile.\n")
    for f in source_files:
        compile_source(client, f)

    print(f"\n🎉  Wiki compilation complete!")
    print(f"    Wiki pages are in: {WIKI_DIR}/")

if __name__ == "__main__":
    main()
