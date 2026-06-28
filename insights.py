"""
insights.py — Wiki Health & Intelligence Tools
Detects knowledge gaps and ranks sources by contribution.

Usage:
    python insights.py           # run both gap detection and source ranking
    python insights.py --gaps    # gap detection only
    python insights.py --rank    # source ranking only
    python insights.py --json    # output as JSON (used by app.py)
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    from openai import OpenAI
    import time
except ImportError:
    print("❌  openai not installed. Run: pip3 install openai")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent
WIKI_DIR     = BASE_DIR / "wiki"
SOURCES_DIR  = BASE_DIR / "sources"
INDEX_FILE   = WIKI_DIR / "INDEX.md"
COMPILED_LOG = BASE_DIR / ".compiled"
MODEL        = "meta/llama-3.3-70b-instruct"

# ── NVIDIA NIM client ─────────────────────────────────────────────────────────

def get_client():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("❌  NVIDIA_API_KEY not set.")
        print("    Export it first: export NVIDIA_API_KEY=nvapi-xxxx")
        sys.exit(1)
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

def call_api(client: OpenAI, messages: list, max_tokens: int = 2048) -> str:
    for attempt in range(3):
        try:
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
                print(f"   ⚠️  Retrying in 10s... ({e})")
                time.sleep(10)
            else:
                raise

# ── Wiki readers ──────────────────────────────────────────────────────────────

def load_wiki_index() -> str:
    return INDEX_FILE.read_text(encoding="utf-8") if INDEX_FILE.exists() else ""

def load_all_pages() -> dict[str, str]:
    """Returns {stem: content} for all wiki pages."""
    pages = {}
    for f in WIKI_DIR.glob("*.md"):
        if f.name in ("INDEX.md", "CONFLICTS.md", "_raw_output.md"):
            continue
        pages[f.stem] = f.read_text(encoding="utf-8", errors="replace")
    return pages

def load_all_sources() -> dict[str, str]:
    """Returns {filename: content} for all compiled sources."""
    compiled = set()
    if COMPILED_LOG.exists():
        compiled = set(COMPILED_LOG.read_text().splitlines())
    sources = {}
    for f in SOURCES_DIR.glob("*"):
        if f.is_file() and f.suffix in {".txt", ".md"} and f.name in compiled:
            sources[f.name] = f.read_text(encoding="utf-8", errors="replace")
    return sources

# ── Gap Detector ──────────────────────────────────────────────────────────────

def detect_gaps(client: OpenAI) -> dict:
    """
    Analyse the wiki to find:
    1. Concepts mentioned/linked but without their own page
    2. Topics the wiki is thin on based on the domain
    3. Suggested next sources to read
    """
    print("🔍  Analysing wiki for knowledge gaps...")

    pages      = load_all_pages()
    wiki_index = load_wiki_index()

    if not pages:
        return {"error": "Wiki is empty — compile some sources first."}

    # Find all [[links]] across all pages
    all_links = set()
    for content in pages.values():
        found = re.findall(r"\[\[([^\]]+)\]\]", content)
        all_links.update(found)

    # Orphan links — referenced but no page exists
    existing_pages = set(pages.keys())
    orphan_links   = sorted(all_links - existing_pages - {"INDEX"})

    # Page word counts — thin pages
    page_lengths = {
        stem: len(content.split())
        for stem, content in pages.items()
    }
    thin_pages = {
        stem: wc for stem, wc in page_lengths.items()
        if wc < 120  # stub threshold
    }

    # Build page summary for the LLM
    page_summary = "\n".join(
        f"- {stem} ({wc} words)" for stem, wc in sorted(page_lengths.items(), key=lambda x: -x[1])
    )

    # Ask the model for deeper gap analysis
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledge curator analysing a personal wiki. "
                "Based on the wiki's current pages and index, identify:\n"
                "1. Important related topics that are completely missing\n"
                "2. Existing topics that need much more depth\n"
                "3. Specific types of sources the person should read next to fill the gaps\n\n"
                "Be specific and actionable. Format your response as:\n\n"
                "MISSING_TOPICS\n"
                "<bullet list of missing topics>\n"
                "END\n\n"
                "NEEDS_DEPTH\n"
                "<bullet list of thin topics and why they need more>\n"
                "END\n\n"
                "SUGGESTED_READING\n"
                "<bullet list of specific source types or papers to find>\n"
                "END"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Wiki Index\n{wiki_index}\n\n"
                f"## All Pages with Word Counts\n{page_summary}\n\n"
                f"## Orphan Links (referenced but no page exists)\n"
                + ("\n".join(f"- {l}" for l in orphan_links) if orphan_links else "None")
            ),
        },
    ]

    raw = call_api(client, messages, max_tokens=1500)

    # Parse sections
    def extract_section(text, tag):
        match = re.search(rf"{tag}\n(.*?)END", text, re.DOTALL)
        if not match:
            return []
        lines = [
            l.strip().lstrip("-•* ").strip()
            for l in match.group(1).strip().splitlines()
            if l.strip() and l.strip() not in ("-", "•", "*")
        ]
        return lines

    return {
        "orphan_links":     orphan_links,
        "thin_pages":       thin_pages,
        "missing_topics":   extract_section(raw, "MISSING_TOPICS"),
        "needs_depth":      extract_section(raw, "NEEDS_DEPTH"),
        "suggested_reading":extract_section(raw, "SUGGESTED_READING"),
        "page_count":       len(pages),
        "total_words":      sum(page_lengths.values()),
    }

# ── Source Ranker ─────────────────────────────────────────────────────────────

def rank_sources(client: OpenAI) -> dict:
    """
    Score each compiled source by how much it contributed to the wiki.
    Signals: pages created, pages updated, links introduced, word delta.
    """
    print("📊  Ranking sources by contribution...")

    pages   = load_all_pages()
    sources = load_all_sources()

    if not sources:
        return {"error": "No compiled sources found."}

    if not pages:
        return {"error": "Wiki is empty — nothing to rank against."}

    # For each source, count how many wiki pages cite it
    scores = []
    for source_name, source_content in sources.items():
        # Count wiki pages that mention this source in their Sources: field
        citation_count = sum(
            1 for content in pages.values()
            if source_name in content or Path(source_name).stem in content
        )

        # Count [[links]] introduced in the source text
        links_in_source = len(re.findall(r"\[\[([^\]]+)\]\]", source_content))

        # Word count of source
        word_count = len(source_content.split())

        # Composite score: citations weighted most heavily
        score = (citation_count * 10) + (links_in_source * 2) + min(word_count // 100, 5)

        # Extract title from source (first # heading or filename)
        title_match = re.search(r"^#\s+(.+)$", source_content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else Path(source_name).stem

        # Extract source URL if clipped from web
        url_match = re.search(r"\*\*Source URL:\*\*\s*(.+)", source_content)
        url = url_match.group(1).strip() if url_match else None

        scores.append({
            "filename":       source_name,
            "title":          title[:60],
            "url":            url,
            "citation_count": citation_count,
            "word_count":     word_count,
            "score":          score,
        })

    # Sort by score descending
    scores.sort(key=lambda x: -x["score"])

    # Ask the LLM to add qualitative insight on top
    top_sources = "\n".join(
        f"{i+1}. {s['title']} (score: {s['score']}, cited by {s['citation_count']} pages)"
        for i, s in enumerate(scores[:10])
    )
    low_sources = "\n".join(
        f"- {s['title']} (score: {s['score']})"
        for s in scores if s["score"] == 0
    )

    wiki_index = load_wiki_index()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledge curator. Given a ranked list of sources and a wiki index, "
                "provide brief qualitative insight:\n"
                "1. Why the top sources likely contributed so much\n"
                "2. What the low-scoring sources might be missing or why they added little\n"
                "3. One recommendation for the knowledge base going forward\n\n"
                "Keep it concise — 3 short paragraphs max."
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Wiki Index\n{wiki_index}\n\n"
                f"## Top Sources\n{top_sources}\n\n"
                f"## Low/Zero Scoring Sources\n{low_sources if low_sources else 'None'}"
            ),
        },
    ]

    insight = call_api(client, messages, max_tokens=600)

    return {
        "ranked_sources": scores,
        "insight":        insight,
        "total_sources":  len(scores),
    }

# ── Pretty printers ───────────────────────────────────────────────────────────

def print_gaps(gaps: dict):
    if "error" in gaps:
        print(f"❌  {gaps['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  KNOWLEDGE GAP REPORT")
    print(f"  {gaps['page_count']} pages · {gaps['total_words']:,} total words")
    print(f"{'='*60}\n")

    if gaps["orphan_links"]:
        print(f"🔗  ORPHAN LINKS ({len(gaps['orphan_links'])} referenced but no page):")
        for l in gaps["orphan_links"]:
            print(f"    • {l}")
        print()

    if gaps["thin_pages"]:
        print(f"📄  THIN PAGES (stub-level, need more content):")
        for stem, wc in sorted(gaps["thin_pages"].items(), key=lambda x: x[1]):
            print(f"    • {stem} ({wc} words)")
        print()

    if gaps["missing_topics"]:
        print(f"❓  MISSING TOPICS:")
        for t in gaps["missing_topics"]:
            print(f"    • {t}")
        print()

    if gaps["needs_depth"]:
        print(f"📈  NEEDS MORE DEPTH:")
        for t in gaps["needs_depth"]:
            print(f"    • {t}")
        print()

    if gaps["suggested_reading"]:
        print(f"📚  SUGGESTED READING:")
        for t in gaps["suggested_reading"]:
            print(f"    • {t}")
        print()

def print_ranking(ranking: dict):
    if "error" in ranking:
        print(f"❌  {ranking['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  SOURCE RANKING REPORT")
    print(f"  {ranking['total_sources']} sources compiled")
    print(f"{'='*60}\n")

    for i, s in enumerate(ranking["ranked_sources"]):
        bar   = "█" * min(s["score"], 20)
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1:2}."
        print(f"  {medal}  {s['title']}")
        print(f"       Score: {s['score']}  |  {bar}")
        print(f"       Cited by {s['citation_count']} wiki pages · {s['word_count']:,} words")
        if s.get("url"):
            print(f"       {s['url']}")
        print()

    print(f"💡  INSIGHT:\n")
    for line in ranking["insight"].strip().splitlines():
        print(f"    {line}")
    print()

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Wiki Insights")
    parser.add_argument("--gaps", action="store_true", help="Run gap detection")
    parser.add_argument("--rank", action="store_true", help="Run source ranking")
    parser.add_argument("--json", action="store_true", help="Output as JSON (for app.py)")
    args = parser.parse_args()

    # Default: run both
    run_gaps = args.gaps or (not args.gaps and not args.rank)
    run_rank = args.rank or (not args.gaps and not args.rank)

    client = get_client()
    output = {}

    if run_gaps:
        gaps = detect_gaps(client)
        output["gaps"] = gaps
        if not args.json:
            print_gaps(gaps)

    if run_rank:
        ranking = rank_sources(client)
        output["ranking"] = ranking
        if not args.json:
            print_ranking(ranking)

    if args.json:
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
