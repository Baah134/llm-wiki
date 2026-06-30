"""
clipper.py — Web Clipper + PDF + YouTube + Batch Ingester for LLM Wiki
Handles URLs, PDFs, YouTube videos, and batch files.

Usage:
    python clipper.py https://article.com              # clip a web article
    python clipper.py https://youtube.com/watch?v=xxx  # extract YouTube transcript
    python clipper.py paper.pdf                         # ingest a PDF
    python clipper.py --batch urls.txt                  # clip multiple URLs from file
    python clipper.py https://... --no-compile          # save but don't compile yet
    python clipper.py https://... --force               # re-clip even if already done
    python clipper.py https://... --preview             # preview content before compiling
    python clipper.py --list                            # show all clipped sources
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs

try:
    import trafilatura
except ImportError:
    print("❌  trafilatura not installed. Run: pip3 install trafilatura")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("❌  tqdm not installed. Run: pip3 install tqdm")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
SOURCES_DIR = BASE_DIR / "sources"
CLIPS_LOG   = BASE_DIR / ".clipped"

# ── Log helpers ───────────────────────────────────────────────────────────────

def get_clipped() -> dict:
    """Returns {source_key: filename} of already clipped sources."""
    if not CLIPS_LOG.exists():
        return {}
    result = {}
    for line in CLIPS_LOG.read_text().splitlines():
        if "|" in line:
            key, filename = line.split("|", 1)
            result[key.strip()] = filename.strip()
    return result

def log_clipped(key: str, filename: str):
    with open(CLIPS_LOG, "a") as f:
        f.write(f"{key}|{filename}\n")

def remove_clipped(key: str):
    """Remove a key from the clipped log (for --force re-clip)."""
    if not CLIPS_LOG.exists():
        return
    lines = [
        l for l in CLIPS_LOG.read_text().splitlines()
        if not l.startswith(key)
    ]
    CLIPS_LOG.write_text("\n".join(lines) + "\n")

# ── Filename helpers ──────────────────────────────────────────────────────────

def to_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^\w\-]", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len]

def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path   = parsed.path.strip("/").replace("/", "_")
    slug   = to_slug(f"{domain}_{path}" if path else domain)
    return f"{datetime.now().strftime('%Y%m%d')}_{slug}.txt"

def pdf_to_filename(pdf_path: Path) -> str:
    return f"{datetime.now().strftime('%Y%m%d')}_{to_slug(pdf_path.stem)}.txt"

def youtube_to_filename(video_id: str, title: str) -> str:
    slug = to_slug(title) if title else video_id
    return f"{datetime.now().strftime('%Y%m%d')}_yt_{slug}.txt"

# ── YouTube detection ─────────────────────────────────────────────────────────

def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    return any(host in parsed.netloc for host in ["youtube.com", "youtu.be", "www.youtube.com"])

def get_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("?")[0]
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        ids = qs.get("v", [])
        return ids[0] if ids else None
    return None

# ── Extractors ────────────────────────────────────────────────────────────────

def extract_url(url: str) -> tuple[str, str]:
    """Fetch and extract article text. Returns (title, text)."""
    print(f"   🌐  Fetching: {url}")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Could not fetch URL. Page may be JS-rendered or unavailable.")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    metadata = trafilatura.extract_metadata(downloaded)
    title    = metadata.title if metadata and metadata.title else url

    if not text:
        raise ValueError(
            "Could not extract text. Page may require JavaScript.\n"
            "    💡  Tip: Use the Obsidian Web Clipper browser extension for JS-heavy pages.\n"
            "        Install from: obsidian.md/clipper\n"
            "        Save clipped content into your sources/ folder and the watcher handles the rest."
        )
    return title, text

def extract_youtube(url: str) -> tuple[str, str]:
    """Extract transcript from a YouTube video. Returns (title, text)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        raise ImportError("youtube-transcript-api not installed. Run: pip3 install youtube-transcript-api")

    video_id = get_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    print(f"   🎬  Fetching transcript for video: {video_id}")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prefer manual English, fall back to auto-generated, then any language
        try:
            transcript = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
            except Exception:
                # Take whatever is available
                transcript = next(iter(transcript_list))

        entries   = transcript.fetch()
        lang_code = transcript.language_code
        print(f"   🌐  Language: {transcript.language} ({'auto-generated' if transcript.is_generated else 'manual'})")

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No transcript found for this video.")

    # Build clean readable text from transcript entries
    # Group into paragraphs every ~10 entries for readability
    lines  = []
    buffer = []
    for i, entry in enumerate(entries):
        text = entry.get("text", "").strip()
        text = re.sub(r"\[.*?\]", "", text).strip()  # remove [Music], [Applause] etc.
        if text:
            buffer.append(text)
        if len(buffer) >= 10 or i == len(entries) - 1:
            if buffer:
                lines.append(" ".join(buffer))
                buffer = []

    full_text  = "\n\n".join(lines)
    word_count = len(full_text.split())
    print(f"   📝  Extracted {word_count:,} words from transcript ({len(entries)} segments)")

    # Try to get video title from page metadata
    try:
        downloaded = trafilatura.fetch_url(url)
        metadata   = trafilatura.extract_metadata(downloaded) if downloaded else None
        title      = metadata.title if metadata and metadata.title else f"YouTube Video {video_id}"
    except Exception:
        title = f"YouTube Video {video_id}"

    return title, full_text

def extract_pdf(pdf_path: Path) -> tuple[str, str]:
    """Extract text from a PDF. Returns (title, text)."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip3 install pymupdf")

    print(f"   📄  Reading PDF: {pdf_path.name}")
    doc        = fitz.open(str(pdf_path))
    pages_text = [page.get_text("text") for page in doc if page.get_text("text").strip()]
    doc.close()

    if not pages_text:
        raise ValueError("Could not extract text. This may be a scanned/image PDF.")

    full_text  = "\n\n".join(pages_text)
    word_count = len(full_text.split())
    print(f"   📝  Extracted {word_count:,} words from {len(pages_text)} page(s)")

    title = pdf_path.stem.replace("_", " ").replace("-", " ").title()
    return title, full_text

# ── Savers ────────────────────────────────────────────────────────────────────

def save_source(filename: str, title: str, text: str, meta: dict) -> Path:
    """Save content to sources/ with a metadata header."""
    SOURCES_DIR.mkdir(exist_ok=True)
    filepath = SOURCES_DIR / filename

    meta_lines = "\n".join(f"**{k}:** {v}" for k, v in meta.items())
    content    = f"# {title}\n\n{meta_lines}\n\n---\n\n{text}\n"
    filepath.write_text(content, encoding="utf-8")
    return filepath

# ── Preview ───────────────────────────────────────────────────────────────────

def preview(title: str, text: str):
    word_count = len(text.split())
    snippet    = " ".join(text.split()[:80])
    print(f"\n   📰  Title:  {title}")
    print(f"   📝  Words:  {word_count:,}")
    print(f"   🔍  Preview: {snippet}...")

# ── Single clip ───────────────────────────────────────────────────────────────

def clip_one(user_input: str, no_compile: bool = False, force: bool = False,
             show_preview: bool = False, silent: bool = False) -> bool:
    """
    Clip a single URL, YouTube link, or PDF.
    Returns True on success, False on failure.
    silent=True suppresses most output (used in batch mode).
    """
    clipped = get_clipped()

    # ── Determine input type ──
    input_path = Path(user_input)
    is_pdf     = input_path.suffix.lower() == ".pdf" and input_path.exists()
    is_yt      = not is_pdf and is_youtube_url(user_input)
    is_url     = not is_pdf and not is_yt

    # Canonical key for dedup tracking
    key = str(input_path.resolve()) if is_pdf else user_input

    # ── Dedup check ──
    if key in clipped and not force:
        existing = clipped[key]
        if not silent:
            print(f"⚠️  Already clipped: {existing}")
            print(f"   Use --force to re-clip.")
        return False

    if force and key in clipped:
        remove_clipped(key)
        old_file = SOURCES_DIR / clipped[key]
        if old_file.exists():
            old_file.unlink()
        if not silent:
            print(f"   🔄  Force re-clipping...")

    # ── Extract ──
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if is_pdf:
            title, text = extract_pdf(input_path)
            filename    = pdf_to_filename(input_path)
            meta        = {"Source": f"{input_path.name} (PDF)", "Ingested": timestamp}

        elif is_yt:
            title, text = extract_youtube(user_input)
            video_id    = get_video_id(user_input) or "unknown"
            filename    = youtube_to_filename(video_id, title)
            meta        = {"Source": f"YouTube: {user_input}", "Clipped": timestamp, "Video ID": video_id}

        else:
            title, text = extract_url(user_input)
            filename    = url_to_filename(user_input)
            meta        = {"Source URL": user_input, "Clipped": timestamp}

    except Exception as e:
        if not silent:
            print(f"❌  Failed: {e}")
        else:
            tqdm.write(f"   ❌  Failed: {user_input[:60]} — {str(e)[:80]}")
        return False

    # ── Preview ──
    if show_preview:
        preview(title, text)
        confirm = input("\n   Compile this? (y/n): ").strip().lower()
        if confirm != "y":
            print("   ⏭️  Skipped.")
            return False

    # ── Save ──
    filepath = save_source(filename, title, text, meta)
    log_clipped(key, filename)

    if not silent:
        print(f"   💾  Saved to: sources/{filename}")

    # ── Compile ──
    if not no_compile:
        if not silent:
            print(f"\n   🚀  Running compiler...\n")
        result = os.system(f'python3 "{BASE_DIR}/compiler.py" --file "{filename}"')
        if result != 0 and not silent:
            print(f"   ⚠️  Compiler exited with errors. Check output above.")

    return True

# ── Batch clip ────────────────────────────────────────────────────────────────

def clip_batch(batch_file: Path, no_compile: bool = False, force: bool = False):
    """Clip all URLs/paths listed in a text file, one per line."""
    if not batch_file.exists():
        print(f"❌  Batch file not found: {batch_file}")
        sys.exit(1)

    lines = [
        l.strip() for l in batch_file.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]

    if not lines:
        print("❌  Batch file is empty.")
        sys.exit(1)

    print(f"\n📦  Batch clipping {len(lines)} source(s)...\n")

    succeeded = 0
    failed    = 0
    skipped   = 0

    with tqdm(lines, desc="Clipping", unit="source", ncols=80) as bar:
        for item in bar:
            bar.set_description(f"Clipping: {item[:40]}...")
            clipped = get_clipped()
            key     = str(Path(item).resolve()) if Path(item).exists() else item

            if key in clipped and not force:
                tqdm.write(f"   ⏭️  Already clipped: {item[:60]}")
                skipped += 1
                bar.update()
                continue

            ok = clip_one(item, no_compile=no_compile, force=force, silent=True)
            if ok:
                tqdm.write(f"   ✅  {item[:70]}")
                succeeded += 1
            else:
                tqdm.write(f"   ❌  {item[:70]}")
                failed += 1

    print(f"\n📊  Batch complete:")
    print(f"    ✅  {succeeded} succeeded")
    print(f"    ⏭️  {skipped} skipped (already clipped)")
    print(f"    ❌  {failed} failed")

    if not no_compile and succeeded > 0:
        print(f"\n🚀  Running compiler on all new sources...\n")
        os.system(f'python3 "{BASE_DIR}/compiler.py"')

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Wiki Clipper — URLs, YouTube, PDFs, batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clipper.py https://article.com
  python clipper.py https://youtube.com/watch?v=dQw4w9WgXcQ
  python clipper.py paper.pdf
  python clipper.py --batch urls.txt
  python clipper.py https://article.com --force
  python clipper.py https://article.com --preview
  python clipper.py --list
        """
    )
    parser.add_argument("input",       nargs="?", help="URL, YouTube link, or path to PDF")
    parser.add_argument("--batch",     type=str,  help="Path to a text file with one URL/path per line")
    parser.add_argument("--no-compile",action="store_true", help="Save to sources/ but don't compile yet")
    parser.add_argument("--force",     action="store_true", help="Re-clip even if already clipped")
    parser.add_argument("--preview",   action="store_true", help="Preview content before compiling")
    parser.add_argument("--list",      action="store_true", help="List all previously clipped sources")
    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        clipped = get_clipped()
        if not clipped:
            print("No sources clipped yet.")
        else:
            print(f"\n{'Source':<65} {'File'}")
            print("─" * 100)
            for key, fname in clipped.items():
                print(f"{key[:65]:<65} {fname}")
            print(f"\nTotal: {len(clipped)} source(s)")
        return

    # ── Batch mode ──
    if args.batch:
        clip_batch(
            Path(args.batch),
            no_compile=args.no_compile,
            force=args.force,
        )
        return

    # ── Single mode ──
    if not args.input:
        parser.print_help()
        sys.exit(1)

    user_input = args.input.strip()
    input_type = "YouTube video" if is_youtube_url(user_input) else \
                 "PDF" if Path(user_input).suffix.lower() == ".pdf" else "article"

    print(f"\n📎  Clipping {input_type}: {user_input}\n")

    ok = clip_one(
        user_input,
        no_compile=args.no_compile,
        force=args.force,
        show_preview=args.preview,
    )

    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
