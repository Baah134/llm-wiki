"""
watcher.py — Auto-compiler watcher for LLM Wiki
Monitors sources/ and compiles new files automatically.

Usage:
    python watcher.py          # start watching (runs until Ctrl+C)
    python watcher.py --once   # process any pending files and exit
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("❌  watchdog not installed.")
    print("    Run: pip3 install watchdog")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent
SOURCES_DIR  = BASE_DIR / "sources"
COMPILED_LOG = BASE_DIR / ".compiled"
SUPPORTED    = {".txt", ".md", ".pdf"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_compiled_files() -> set:
    if not COMPILED_LOG.exists():
        return set()
    return set(COMPILED_LOG.read_text().splitlines())

def compile_file(filepath: Path):
    """Run the compiler on a single file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] 📄  New file detected: {filepath.name}")
    print(f"[{timestamp}] 🚀  Compiling...\n")
    os.system(f'python3 "{BASE_DIR}/compiler.py" --file "{filepath.name}"')

def process_pending():
    """Compile any files in sources/ that haven't been compiled yet."""
    compiled = get_compiled_files()
    pending = [
        f for f in sorted(SOURCES_DIR.iterdir())
        if f.is_file() and f.suffix in SUPPORTED and f.name not in compiled
    ]
    if not pending:
        print("✅  No pending files.")
        return
    print(f"📚  Found {len(pending)} pending file(s).\n")
    for f in pending:
        compile_file(f)

# ── Watchdog event handler ────────────────────────────────────────────────────

class SourceHandler(FileSystemEventHandler):
    def __init__(self):
        self._processing = set()  # debounce — avoid double-triggers

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED:
            return
        if path.name in self._processing:
            return

        # Small delay — wait for file to finish writing before reading
        time.sleep(1.5)

        # Check it's not already compiled
        compiled = get_compiled_files()
        if path.name in compiled:
            return

        self._processing.add(path.name)
        try:
            compile_file(path)
        finally:
            self._processing.discard(path.name)

    def on_moved(self, event):
        """Also handle files moved/dragged into the sources/ folder."""
        if event.is_directory:
            return
        path = Path(event.dest_path)
        if path.suffix.lower() not in SUPPORTED:
            return

        time.sleep(1.5)

        compiled = get_compiled_files()
        if path.name in compiled:
            return

        if path.name in self._processing:
            return

        self._processing.add(path.name)
        try:
            compile_file(path)
        finally:
            self._processing.discard(path.name)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Wiki Watcher")
    parser.add_argument("--once", action="store_true", help="Process pending files and exit")
    args = parser.parse_args()

    SOURCES_DIR.mkdir(exist_ok=True)

    if args.once:
        process_pending()
        return

    # Start watching
    print(f"👁️   Watching: {SOURCES_DIR}")
    print(f"    Supported file types: {', '.join(SUPPORTED)}")
    print(f"    Drop files into sources/ to compile automatically.")
    print(f"    Press Ctrl+C to stop.\n")

    # Process any files that came in while the watcher wasn't running
    process_pending()

    handler  = SourceHandler()
    observer = Observer()
    observer.schedule(handler, str(SOURCES_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋  Watcher stopped.")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
