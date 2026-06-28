"""
clipper.py — Web Clipper + PDF Ingester for LLM Wiki
Handles URLs and PDF files — saves to sources/ and compiles into the wiki.

Usage:
    python clipper.py https://example.com/article     # clip a web page
    python clipper.py paper.pdf                        # ingest a PDF
    python clipper.py https://... --no-compile         # save but don't compile yet
    python clipper.py --list                           # show all clipped sources
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

try:
    import trafilatura
except ImportError:
    print("❌  trafilatura not installed.")
    print("    Run: pip3 install trafilatura")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
SOURCES_DIR = BASE_DIR / "sources"
CLIPS_LOG   = BASE_DIR / ".clipped"   # tracks clipped URLs

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_clipped_urls() -> dict:
    """Returns {url: filename} of already clipped URLs."""
    if not CLIPS_LOG.exists():
        return {}
    result = {}
    for line in CLIPS_LOG.read_text().splitlines():
        if "|" in line:
            url, filename = line.split("|", 1)
            result[url.strip()] = filename.strip()
    return result

def log_clipped(url: str, filename: str):
    with open(CLIPS_LOG, "a") as f:
        f.write(f"{url}|{filename}\n")

def url_to_filename(url: str) -> str:
    """Turn a URL into a clean readable filename."""
    parsed = urlparse(url)
    # Use domain + path, strip trailing slashes
    domain = parsed.netloc.replace("www.", "")
    path   = parsed.path.strip("/").replace("/", "_")
    slug   = f"{domain}_{path}" if path else domain
    # Remove anything that's not alphanumeric, dash, or underscore
    slug   = re.sub(r"[^\w\-]", "_", slug)
    slug   = re.sub(r"_+", "_", slug).strip("_")
    # Truncate if too long
    slug   = slug[:80]
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"{timestamp}_{slug}.txt"

def fetch_and_extract(url: str) -> tuple[str, str]:
    """
    Fetch URL and extract clean text using trafilatura.
    Returns (title, clean_text).
    """
    print(f"   🌐  Fetching: {url}")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print("❌  Could not fetch the URL. Check your connection or try another URL.")
        sys.exit(1)

    # Extract clean article text
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    # Extract metadata for the title
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else url

    if not text:
        print("❌  Could not extract article text from this page.")
        print("    The page may require JavaScript or be behind a login.")
        sys.exit(1)

    return title, text

def save_to_sources(url: str, title: str, text: str) -> Path:
    """Save extracted text to sources/ with metadata header."""
    SOURCES_DIR.mkdir(exist_ok=True)
    filename = url_to_filename(url)
    filepath = SOURCES_DIR / filename

    # Add a metadata header so the compiler knows the provenance
    content = f"""# {title}

**Source URL:** {url}
**Clipped:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

{text}
"""
    filepath.write_text(content, encoding="utf-8")
    return filepath

def pdf_to_filename(pdf_path: Path) -> str:
    """Turn a PDF filename into a clean source filename."""
    slug = re.sub(r"[^\w\-]", "_", pdf_path.stem)
    slug = re.sub(r"_+", "_", slug).strip("_")[:80]
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"{timestamp}_{slug}.txt"

def extract_pdf(pdf_path: Path) -> tuple[str, str]:
    """
    Extract clean text from a PDF using PyMuPDF.
    Returns (title, text).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌  PyMuPDF not installed.")
        print("    Run: pip3 install pymupdf")
        sys.exit(1)

    print(f"   📄  Reading PDF: {pdf_path.name}")

    doc = fitz.open(str(pdf_path))
    pages_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)

    doc.close()

    if not pages_text:
        print("❌  Could not extract text from this PDF.")
        print("    It may be a scanned image PDF — OCR support coming later.")
        sys.exit(1)

    full_text = "\n\n".join(pages_text)
    word_count = len(full_text.split())
    print(f"   📝  Extracted {word_count:,} words from {len(pages_text)} page(s)")

    # Use the filename as title, cleaned up
    title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
    return title, full_text

def save_pdf_to_sources(pdf_path: Path, title: str, text: str) -> Path:
    """Save extracted PDF text to sources/ with metadata header."""
    SOURCES_DIR.mkdir(exist_ok=True)
    filename = pdf_to_filename(pdf_path)
    filepath = SOURCES_DIR / filename

    content = f"""# {title}

**Source:** {pdf_path.name} (PDF)
**Ingested:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Pages extracted:** {text.count(chr(12)) + 1}

---

{text}
"""
    filepath.write_text(content, encoding="utf-8")
    return filepath

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Wiki Web Clipper + PDF Ingester")
    parser.add_argument("input", nargs="?", help="URL or path to a PDF file")
    parser.add_argument("--no-compile", action="store_true", help="Save to sources/ but don't compile yet")
    parser.add_argument("--list", action="store_true", help="List all previously clipped sources")
    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        clipped = get_clipped_urls()
        if not clipped:
            print("No sources clipped yet.")
        else:
            print(f"{'Source':<60} {'File'}")
            print("-" * 90)
            for url, fname in clipped.items():
                print(f"{url:<60} {fname}")
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    user_input = args.input.strip()

    # ── Detect: PDF file or URL? ──
    input_path = Path(user_input)
    is_pdf = input_path.suffix.lower() == ".pdf" and input_path.exists()

    if is_pdf:
        # ── PDF mode ──
        print(f"\n📎  Ingesting PDF: {input_path.name}\n")
        title, text = extract_pdf(input_path)
        print(f"   📰  Title: {title}")
        filepath = save_pdf_to_sources(input_path, title, text)
        log_clipped(str(input_path.resolve()), filepath.name)
        print(f"   💾  Saved to: sources/{filepath.name}")

    else:
        # ── URL mode ──
        url = user_input

        # Check if already clipped
        clipped = get_clipped_urls()
        if url in clipped:
            existing = clipped[url]
            print(f"⚠️  Already clipped: {existing}")
            print(f"   Delete {existing} from sources/ and re-run to re-clip.")
            if not args.no_compile:
                print("   Running compiler on existing file...")
                os.system(f'python3 "{BASE_DIR}/compiler.py" --file "{existing}"')
            return

        print(f"\n📎  Clipping: {url}\n")
        title, text = fetch_and_extract(url)
        print(f"   📰  Title: {title}")
        print(f"   📝  Extracted {len(text.split()):,} words")
        filepath = save_to_sources(url, title, text)
        log_clipped(url, filepath.name)
        print(f"   💾  Saved to: sources/{filepath.name}")

    # ── Compile ──
    if args.no_compile:
        print("\n   ⏭️  Skipping compilation (--no-compile). Run compiler.py when ready.")
    else:
        print("\n   🚀  Running compiler...\n")
        os.system(f'python3 "{BASE_DIR}/compiler.py" --file "{filepath.name}"')

if __name__ == "__main__":
    main()
