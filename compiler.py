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

# Hardcoded baselines in case discovery network check fails
DEFAULT_POOL = [
    "deepseek-ai/deepseek-v4-flash",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-70b-instruct"
]
MAX_TOKENS     = 16384      # Increased to give DeepSeek's thinking engine breathing room
CHUNK_CHAR_LIMIT = 60_000   # ~15k tokens of source per chunk

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
    if not name.endswith(".md"):
        name = name + ".md"
    path = WIKI_DIR / name
    if path.exists():
        return f"### FILE: {name}\n\n{path.read_text(encoding='utf-8')}"
    return ""

def load_all_wiki_pages() -> str:
    pages = []
    for f in sorted(WIKI_DIR.glob("*.md")):
        if f.name in ("INDEX.md", "CONFLICTS.md", "_raw_output.md"):
            continue
        pages.append(f"### FILE: {f.name}\n\n{f.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(pages) if pages else "_(No wiki pages exist yet.)_"

def get_compiled_files() -> set:
    if not COMPILED_LOG.exists():
        return set()
    return set(COMPILED_LOG.read_text().splitlines())

def mark_compiled(filename: str):
    with open(COMPILED_LOG, "a") as f:
        f.write(filename + "\n")

def discover_model_pool(client) -> list[str]:
    """
    Queries live NVIDIA endpoints and sorts text models dynamically by:
    Tier 1: Fast variants with large context windows
    Tier 2: Heavyweight variants with large context windows
    Tier 3: Standard fast models (typical context limits)
    Tier 4: Baseline legacy / standard models
    """
    try:
        print("   🔍  Scanning NVIDIA network for active free endpoints...")
        api_models = client.models.list().data
        discovered = []
        
        for item in api_models:
            model_id = item.id.lower()
            # Omit vector embedders, visual networks, audio services, and guard rails
            if any(ignore in model_id for ignore in ["embed", "rerank", "tts", "asr", "guard", "image", "clip", "cosmos"]):
                continue
            discovered.append(item.id)
            
        if not discovered:
            return DEFAULT_POOL

        def ranking_heuristic(model_name: str) -> int:
            name = model_name.lower()
            # Identifiers for low-latency / fast generation throughput
            is_fast = any(k in name for k in ["flash", "mini", "nano", "8b", "3b", "lite"])
            # Identifiers for extensive structural context limits
            has_large_context = any(k in name for k in ["nemotron", "deepseek", "kimi", "glm", "ultra", "super", "pro", "120b", "550b", "397b"])
            
            if is_fast and has_large_context:
                return 0  # Tier 1: Fast + Massive Context Window
            elif has_large_context:
                return 1  # Tier 2: Heavyweight + Massive Context Window
            elif is_fast:
                return 2  # Tier 3: Fast + Baseline Context Window
            return 3      # Tier 4: Standard / Legacy models

        # Sort based on tier priority first, then alphabetically
        discovered.sort(key=lambda x: (ranking_heuristic(x), x.lower()))
        
        print(f"   🚀  Dynamic discovery successful. Initialized {len(discovered)} active models.")
        print("      Primary execution route:")
        for top_model in discovered[:3]:
            print(f"       • {top_model}")
            
        return discovered
        
    except Exception as e:
        print(f"   ⚠️  Discovery network check failed ({e}). Falling back to baseline configuration.")
        return DEFAULT_POOL

def call_api(client: OpenAI, messages: list, max_tokens: int = MAX_TOKENS, label: str = "") -> str:
    """Call the API with an expanded retry pool and exponential backoff safety scaling."""
    model_idx = 0
    total_attempts = 30  #  Expanded to give plenty of room for infrastructure rotation
    last_error = None  
    
    for attempt in range(total_attempts):
        current_model = MODEL_POOL[model_idx % len(MODEL_POOL)]
        model_short_name = current_model.split("/")[-1]
        
        try:
            if label:
                print(f"   🤖  {label} (using {model_short_name})...")
            
            # (Keep your existing dynamic token and parameter tier mapping rules here...)
            if "deepseek" in current_model or "kimi" in current_model:
                extra_body = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
                current_max_tokens = max_tokens
            elif "nemotron" in current_model and any(k in current_model for k in ["super", "ultra", "nano", "30b", "120b", "550b"]):
                extra_body = {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": max_tokens}
                current_max_tokens = max_tokens
            else:
                extra_body = {}
                current_max_tokens = min(max_tokens, 4096)

            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=0.3,
                top_p=0.95,
                max_tokens=current_max_tokens,
                extra_body=extra_body,
                stream=False,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = e  
            error_str = str(e)
            status_code = getattr(e, 'status_code', None)
            
            is_504 = (status_code == 504) or ("504" in error_str) or ("gateway timeout" in error_str.lower())
            is_context_limit = (status_code == 400) or ("context length" in error_str.lower()) or ("badrequesterror" in error_str.lower())
            is_not_found = (status_code == 404) or ("404" in error_str) or ("not found" in error_str.lower())
            is_rate_limit = (status_code == 429) or ("429" in error_str) or ("rate limit" in error_str.lower())
            
            if is_504 or is_context_limit or is_not_found or is_rate_limit:
                model_idx += 1  # Shift to the next model tier layout
                
                # ── EXPONENTIAL BACKOFF CALCULATION ──
                # Attempt 1: 2s delay | Attempt 4: 16s delay | Attempt 6+: Caps at 30s max delay
                sleep_duration = min(2 ** (attempt + 1), 30)
                
                print(f"   ⚠️  Attempt {attempt + 1} failed on {model_short_name}.")
                print(f"      🔄 Infrastructure scaling backoff: Waiting {sleep_duration}s before trying fallback...")
                time.sleep(sleep_duration)
            else:
                if attempt < total_attempts - 1:
                    model_idx += 1
                    time.sleep(5)
                else:
                    raise

    # ── THE EXHAUSTION SHIELD ──
    # If the loop finishes entirely without returning a response string, force an error!
    print("\n❌  [CRITICAL] All available model rotation pathways have been exhausted.")
    print("    NVIDIA's shared public tier is dropping connections across all nodes right now.")
    if 'last_error' in locals() and last_error:
        raise last_error
    raise RuntimeError("API infrastructure pool failed to deliver any text content.")

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

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_CHAR_LIMIT, overlap_size: int = 6000) -> list[str]:
    """
    Split text into chunks of roughly chunk_size characters with a sliding window overlap.
    Splits on paragraph boundaries to preserve context and carries over overlap_size 
    characters worth of preceding paragraphs into the next chunk window.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 accounting for the split boundary \n\n
        
        # If adding this paragraph pushes the current frame past the character threshold
        if current_len + para_len > chunk_size and current:
            # Commit the current built chunk frame to memory
            chunks.append("\n\n".join(current))
            
            # Form the sliding window overlap context frame by walking backward
            overlap_paras = []
            overlap_len = 0
            for old_para in reversed(current):
                old_len = len(old_para) + 2
                if overlap_len + old_len > overlap_size:
                    break
                overlap_paras.insert(0, old_para)
                overlap_len += old_len
            
            # Seed the next compilation frame using the context overlap window
            current = overlap_paras + [para]
            current_len = overlap_len + para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks

# ── Step 1: Index-first ───────────────────────────────────────────────────────

def get_relevant_pages(client: OpenAI, source_name: str, raw_content: str) -> list[str]:
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
            ),
        },
    ]

    result = call_api(client, messages, max_tokens=256, label="Checking index for relevant pages")
    
    #  The Null Guard Rail
    if result is None:
        print("   ⚠️  Index check received a null response from the API pool. Skipping.")
        return []
        
    result = result.strip()  # Now completely safe from AttributeError!

    if result.lower() == "none" or not result:
        return []

    pages = [p.strip().replace("[[", "").replace("]]", "") for p in result.split(",")]
    pages = [p if p.endswith(".md") else p + ".md" for p in pages if p]
    return pages

# ── Step 2: Compile one chunk ─────────────────────────────────────────────────

def compile_chunk(client: OpenAI, source_name: str, chunk: str,
                  chunk_num: int, total_chunks: int, relevant_pages: list[str]) -> list[str]:
    """Compile a single chunk of a source file."""
    agent_schema = load_agent_schema()
    wiki_index   = load_wiki_index()
    today        = datetime.now().strftime("%Y-%m-%d")

    if relevant_pages:
        page_context = "\n\n---\n\n".join(
            load_wiki_page(p) for p in relevant_pages if load_wiki_page(p)
        )
        pages_note = f"Relevant existing pages loaded: {', '.join(relevant_pages)}"
    else:
        page_context = "_(No existing pages are relevant to this source.)_"
        pages_note   = "No existing pages loaded — this source introduces new concepts."

    chunk_note = (
        f"This is chunk {chunk_num} of {total_chunks} from the source file. "
        "Integrate it into the wiki as usual. The wiki already reflects earlier chunks."
        if total_chunks > 1 else ""
    )

    system_prompt = f"""
{agent_schema}

Today's date: {today}

## Current Wiki Index
{wiki_index}

## Relevant Existing Wiki Pages
{page_context}

## Note
{pages_note}
{chunk_note}

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

**Filename:** {source_name}{f' (chunk {chunk_num}/{total_chunks})' if total_chunks > 1 else ''}

**Content:**
{chunk}

Read this carefully. Then:
1. Create or update wiki pages for every significant concept, person, method, or tool mentioned.
2. Cross-link pages using [[wiki-links]].
3. Update wiki/INDEX.md to reflect any new or changed pages.

Output all changed files using the FILE block format.
""".strip()

    label = f"Compiling chunk {chunk_num}/{total_chunks}" if total_chunks > 1 else "Compiling source into wiki"
    response_text = call_api(
        client,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        label=label,
    )
    return save_wiki_output(response_text)

def compile_with_context(client: OpenAI, source_path: Path, relevant_pages: list[str]) -> list[str]:
    """Compile a source, chunking automatically if it exceeds the token limit."""
    raw_content = source_path.read_text(encoding="utf-8", errors="replace")
    chunks      = chunk_text(raw_content)
    total       = len(chunks)

    if total > 1:
        print(f"   ✂️   File is large — splitting into {total} chunks (~{len(raw_content):,} chars total)")

    all_saved = []
    for i, chunk in enumerate(chunks, 1):
        saved = compile_chunk(client, source_path.name, chunk, i, total, relevant_pages)
        all_saved.extend(saved)
        # Reload relevant pages after each chunk so the next chunk sees updated wiki
        if i < total:
            relevant_pages = get_relevant_pages(client, source_path.name, chunk)

    return all_saved

# ── Step 3: Stub filler ───────────────────────────────────────────────────────

def fill_missing_stubs(client: OpenAI, saved_pages: list[str]):
    all_links = set()
    for filepath in saved_pages:
        content = (BASE_DIR / filepath).read_text(encoding="utf-8")
        found = re.findall(r"\[\[([^\]]+)\]\]", content)
        all_links.update(found)

    existing = {f.stem for f in WIKI_DIR.glob("*.md")}
    missing  = [link for link in all_links if link not in existing and link != "INDEX"]

    if not missing:
        print("   ✅  No missing linked pages — wiki is consistent.")
        return

    print(f"   🔍  Found {len(missing)} unresolved link(s): {', '.join(missing)}")
    wiki_index = load_wiki_index()

    system_prompt = """
You are a wiki maintainer. Create brief stub pages for concepts that are referenced
but don't have their own page yet. Each stub should have:
- A one-paragraph summary of what the concept is
- A "Related" field with wiki-links to connected concepts
- An "Open Questions" section noting what needs to be filled in later

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
    if not relevant_pages:
        return []

    raw_content  = source_path.read_text(encoding="utf-8", errors="replace")
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

    conflicts = []
    blocks = re.findall(r"CONFLICT\n(.*?)END", result, re.DOTALL)
    for block in blocks:
        page   = re.search(r"Page:\s*(.+)", block)
        wiki   = re.search(r"Wiki says:\s*(.+)", block)
        source = re.search(r"Source says:\s*(.+)", block)
        if page and wiki and source:
            conflicts.append({
                "page":   page.group(1).strip(),
                "wiki":   wiki.group(1).strip(),
                "source": source.group(1).strip(),
            })
    return conflicts

def save_conflicts(source_name: str, conflicts: list[dict]):
    if not conflicts:
        return
    log_path  = WIKI_DIR / "CONFLICTS.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n## {source_name} — {timestamp}\n"]
    for c in conflicts:
        lines.append(f"- **{c['page']}**")
        lines.append(f"  - Wiki says: {c['wiki']}")
        lines.append(f"  - Source says: {c['source']}\n")
    with open(log_path, "a") as f:
        f.write("\n".join(lines))

# ── Step 5: Lint ──────────────────────────────────────────────────────────────

REQUIRED_SECTIONS = ["## Summary", "## Key Details", "## Connections", "## Open Questions"]
SKIP_LINT         = {"INDEX.md", "CONFLICTS.md", "_raw_output.md"}

def run_lint(verbose: bool = True) -> dict:
    """
    Audit the wiki for structural problems.
    Auto-fixes mechanical issues. Reports everything else.
    Returns a dict of findings.
    """
    if verbose:
        print("\n🔍  Running lint...\n")

    pages        = {f.name: f for f in WIKI_DIR.glob("*.md") if f.name not in SKIP_LINT}
    index_text   = load_wiki_index()
    findings     = {
        "missing_sections":  [],   # pages missing required sections
        "orphan_links":      [],   # [[links]] with no target page
        "index_mismatches":  [],   # pages on disk not in INDEX.md (auto-fixed)
        "empty_pages":       [],   # pages with no real content
        "long_pages":        [],   # pages over 1000 words (suggest summarising)
    }

    # ── Collect all [[links]] across all pages ──
    all_links = set()
    for name, path in pages.items():
        content = path.read_text(encoding="utf-8", errors="replace")
        all_links.update(re.findall(r"\[\[([^\]]+)\]\]", content))

    existing_stems = {Path(n).stem for n in pages}

    for name, path in sorted(pages.items()):
        content    = path.read_text(encoding="utf-8", errors="replace")
        word_count = len(content.split())
        stem       = path.stem

        # Missing required sections
        missing = [s for s in REQUIRED_SECTIONS if s not in content]
        if missing:
            findings["missing_sections"].append({
                "page": name, "missing": missing
            })

        # Empty pages
        if word_count < 20:
            findings["empty_pages"].append(name)

        # Long pages
        if word_count > 1000:
            findings["long_pages"].append({"page": name, "words": word_count})

        # Index mismatch — page exists but not in INDEX.md (auto-fix)
        if stem not in index_text and name != "INDEX.md":
            findings["index_mismatches"].append(name)

    # Orphan links — referenced but no page exists
    orphans = sorted(all_links - existing_stems - {"INDEX"})
    findings["orphan_links"] = orphans

    # ── Auto-fix: add missing pages to INDEX.md ──
    if findings["index_mismatches"]:
        _fix_index_mismatches(findings["index_mismatches"])

    # ── Report ──
    if verbose:
        _print_lint_report(findings)

    return findings

def _fix_index_mismatches(missing_from_index: list[str]):
    """Add pages that exist on disk but are missing from INDEX.md."""
    if not INDEX_FILE.exists():
        return
    index_text = INDEX_FILE.read_text(encoding="utf-8")
    additions  = []
    for name in missing_from_index:
        stem = Path(name).stem
        additions.append(f"| [[{stem}]] | — | _(added by lint)_ |")
    updated = index_text.rstrip() + "\n" + "\n".join(additions) + "\n"
    INDEX_FILE.write_text(updated, encoding="utf-8")
    print(f"   🔧  Auto-fixed: added {len(missing_from_index)} missing page(s) to INDEX.md")

def _print_lint_report(findings: dict):
    total_issues = (
        len(findings["missing_sections"]) +
        len(findings["orphan_links"]) +
        len(findings["empty_pages"]) +
        len(findings["long_pages"])
    )

    if total_issues == 0 and not findings["index_mismatches"]:
        print("   ✅  Wiki is clean — no issues found.\n")
        return

    print(f"   Found {total_issues} issue(s):\n")

    if findings["empty_pages"]:
        print(f"   🗑️   Empty pages ({len(findings['empty_pages'])}):")
        for p in findings["empty_pages"]:
            print(f"        • {p} — no content, consider deleting or filling")
        print()

    if findings["missing_sections"]:
        print(f"   📋  Missing required sections ({len(findings['missing_sections'])}):")
        for item in findings["missing_sections"]:
            print(f"        • {item['page']} — missing: {', '.join(item['missing'])}")
        print()

    if findings["orphan_links"]:
        print(f"   🔗  Orphan links — referenced but no page exists ({len(findings['orphan_links'])}):")
        for link in findings["orphan_links"]:
            print(f"        • [[{link}]]")
        print()

    if findings["long_pages"]:
        print(f"   📏  Long pages — consider summarising ({len(findings['long_pages'])}):")
        for item in findings["long_pages"]:
            print(f"        • {item['page']} ({item['words']:,} words)")
        print()

    if findings["index_mismatches"]:
        print(f"   🔧  Auto-fixed: {len(findings['index_mismatches'])} page(s) added to INDEX.md")
        print()

# ── Orchestrator ──────────────────────────────────────────────────────────────

def compile_source(client: OpenAI, source_path: Path) -> list[dict]:
    print(f"\n📄  Compiling: {source_path.name}")

    raw_content  = source_path.read_text(encoding="utf-8", errors="replace")
    conflicts = []
    char_count   = len(raw_content)

    if char_count > CHUNK_CHAR_LIMIT:
        chunks_needed = (char_count // CHUNK_CHAR_LIMIT) + 1
        print(f"   📏  Large file detected ({char_count:,} chars ≈ {char_count//4:,} tokens)")
        print(f"       Will split into ~{chunks_needed} chunks automatically")

    # Step 1 — index-first
    relevant_pages = get_relevant_pages(client, source_path.name, raw_content)
    if relevant_pages:
        print(f"   📎  Relevant pages: {', '.join(relevant_pages)}")
    else:
        print("   📎  No existing pages relevant — starting fresh.")

    

    # Step 3 — compile (with auto-chunking for large files)
    saved = compile_with_context(client, source_path, relevant_pages)

    # Step 4 — fill missing stubs
    if saved:
        fill_missing_stubs(client, saved)

    # Step 5 — lint
    if saved:
        run_lint(verbose=True)

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
    parser.add_argument("--lint",  action="store_true", help="Run lint check only")
    args = parser.parse_args()

    WIKI_DIR.mkdir(exist_ok=True)

    if args.lint:
        run_lint(verbose=True)
        return

    client = get_client()

    # Dynamically inject the full ranked pool list into the script's global runtime scope
    global MODEL_POOL
    MODEL_POOL = discover_model_pool(client)

    if args.query:
        query_wiki(client, args.query)
        return

    # ── 1. Look for raw PDFs and let clipper.py handle them safely ──
    compiled = get_compiled_files() if not args.all else set()
    raw_pdfs = [
        f for f in SOURCES_DIR.iterdir() 
        if f.is_file() and f.suffix.lower() == ".pdf" and f.name not in compiled
    ]

    for pdf in raw_pdfs:
        # Check if clipper.py has already generated a clean text file for this PDF (.endswith safeguard)
        clean_name_slug = re.sub(r"[^\w\-]", "_", pdf.stem).strip("_")[:80]
        already_extracted = any(f.name.endswith(f"{clean_name_slug}.txt") for f in SOURCES_DIR.iterdir())
        
        if already_extracted:
            print(f"   ℹ️  {pdf.name} already has an extracted text file. Skipping local extraction.")
            mark_compiled(pdf.name)
            continue

        print(f"\n🔄 Found raw PDF in batch: {pdf.name}")
        print(f"   Calling clipper.py to extract clean text stream...")
        # Run clipper on the PDF, telling it NOT to compile yet to avoid duplicate loops
        os.system(f'python3 "{BASE_DIR}/clipper.py" "{pdf}" --no-compile')
        # Mark the raw binary PDF as handled so the script skips it next time
        mark_compiled(pdf.name)

    # ── 2. Refresh the compilation log and gather the clean text files ──
    compiled = get_compiled_files() if not args.all else set()
    source_files = [
        f for f in sorted(SOURCES_DIR.iterdir())
        if f.is_file() and f.suffix in {".txt", ".md"} and f.name not in compiled
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