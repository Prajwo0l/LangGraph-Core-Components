"""
main.py — CLI entry point for the Document Summarization System.

Loads environment variables from:
  1. DocumentSummarizer/.env  (if it exists)
  2. LangGraph-Core-Components/.env  (parent folder — shared project .env)

Usage examples
──────────────
# Summarize the entire document
python main.py --file sample_inputs/sample.txt

# Summarize a specific chapter / section by name
python main.py --file sample_inputs/sample.txt --chapter "Chapter 3"
python main.py --file thesis.pdf --chapter "Introduction"
python main.py --file report.pdf --chapter "Conclusion"

# Summarize specific pages (PDF only, 1-based inclusive)
python main.py --file book.pdf --pages 10-25
python main.py --file paper.pdf --pages 1-5

# Combine chapter + depth + mode
python main.py --file thesis.pdf --chapter "Literature Review" --depth detailed --mode bullet

# Save output to JSON
python main.py --file paper.pdf --pages 1-10 --output pages_1_10.json

# Print only the final summary text
python main.py --file paper.pdf --chapter "Results" --plain
"""

import argparse
import json
import sys
from pathlib import Path

# ── Load .env (local first, then parent folder) ───────────────────────────────
from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
_LOCAL_ENV = _THIS_DIR / ".env"
_PARENT_ENV = _THIS_DIR.parent / ".env"

if _LOCAL_ENV.exists():
    load_dotenv(_LOCAL_ENV, override=False)
    _env_source = str(_LOCAL_ENV)
elif _PARENT_ENV.exists():
    load_dotenv(_PARENT_ENV, override=False)
    _env_source = str(_PARENT_ENV)
else:
    _env_source = "none found"

# ─────────────────────────────────────────────────────────────────────────────

from config import Config
from orchestrator.graph import run_pipeline
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="summarizer",
        description="Hierarchical Multi-Agent Document Summarization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Full document
  python main.py --file report.pdf

  # Single chapter (by name)
  python main.py --file thesis.pdf --chapter "Chapter 2"
  python main.py --file notes.txt  --chapter "Introduction"

  # Page range  (PDF only, 1-based inclusive)
  python main.py --file book.pdf --pages 10-25
  python main.py --file paper.pdf --pages 1-5

  # Chapter + detailed bullet summary
  python main.py --file thesis.pdf --chapter "Results" --depth detailed --mode bullet
        """,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the input document (.pdf or .txt)",
    )

    # ── Scope (what part of the document to summarize) ────────────────────────
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--chapter", "-c",
        default=None,
        metavar="NAME",
        help=(
            'Summarize only this chapter/section. '
            'Matched by heading name, e.g. --chapter "Chapter 3" or --chapter "Introduction". '
            'Works for both PDF and TXT.'
        ),
    )
    scope.add_argument(
        "--pages", "-p",
        default=None,
        metavar="START-END",
        help=(
            'Summarize only these pages (PDF only). '
            'Format: START-END  e.g. --pages 10-25  (1-based, inclusive).'
        ),
    )

    # ── Summarization style ───────────────────────────────────────────────────
    parser.add_argument(
        "--depth", "-d",
        choices=["short", "medium", "detailed"],
        default="medium",
        help="Summary depth (default: medium)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["bullet", "paragraph"],
        default="paragraph",
        help="Output mode (default: paragraph)",
    )

    # ── Model / performance ───────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Max parallel worker agents (default: 4)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1500,
        help="Max tokens per chunk (default: 1500)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save full JSON result to this file path",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print only the final summary text (no metadata)",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args()


def _parse_pages(pages_str: str) -> tuple[int, int]:
    """Parse '10-25' → (10, 25). Raises SystemExit on bad format."""
    parts = pages_str.strip().split("-")
    if len(parts) != 2:
        print(f"ERROR: --pages must be in START-END format, e.g. --pages 10-25 (got: {pages_str!r})")
        sys.exit(1)
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        print(f"ERROR: --pages values must be integers (got: {pages_str!r})")
        sys.exit(1)
    if start < 1 or end < start:
        print(f"ERROR: --pages range invalid: start must be ≥1 and end ≥ start (got: {pages_str!r})")
        sys.exit(1)
    return (start, end)


def main() -> None:
    args = parse_args()

    # Build config from CLI arguments
    cfg = Config(
        model=args.model,
        summary_depth=args.depth,
        output_mode=args.mode,
        max_workers=args.workers,
        max_chunk_tokens=args.chunk_tokens,
        log_level=args.log_level,
    )

    log = get_logger("main", cfg)

    # ── Resolve scope ─────────────────────────────────────────────────────────
    page_range: tuple[int, int] | None = None
    chapter: str | None = None

    if args.pages:
        page_range = _parse_pages(args.pages)
    if args.chapter:
        chapter = args.chapter

    # ── Log startup info ──────────────────────────────────────────────────────
    scope_desc = "entire document"
    if page_range:
        scope_desc = f"pages {page_range[0]}–{page_range[1]}"
    elif chapter:
        scope_desc = f"chapter '{chapter}'"

    log.info("Starting summarization pipeline …")
    log.info("  File   : %s", args.file)
    log.info("  Scope  : %s", scope_desc)
    log.info("  Depth  : %s | Mode: %s | Model: %s", args.depth, args.mode, args.model)
    log.info("  .env   : %s", _env_source)

    # ── Validate file ─────────────────────────────────────────────────────────
    if not Path(args.file).exists():
        log.error("File not found: %s", args.file)
        sys.exit(1)

    if page_range and not args.file.lower().endswith(".pdf"):
        log.error("--pages is only supported for PDF files.")
        sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    result = run_pipeline(
        file_path=args.file,
        cfg=cfg,
        page_range=page_range,
        chapter=chapter,
    )

    # ── Print output ──────────────────────────────────────────────────────────
    if args.plain:
        print("\n" + "=" * 70)
        print(f"  {result.get('title', 'Summary')}  [{scope_desc}]")
        print("=" * 70)
        print(result.get("final_summary", "(No summary generated)"))
        print()
    else:
        _print_pretty(result, scope_desc)

    # ── Save to file if requested ─────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Full result saved to: %s", out_path)


def _print_pretty(result: dict, scope_desc: str = "entire document") -> None:
    """Print a nicely formatted version of the result to stdout."""
    meta = result.get("_meta", {})
    title = result.get("title", "Untitled Document")
    section_summaries = result.get("section_summaries", [])
    final_summary = result.get("final_summary", "")

    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  DOCUMENT SUMMARY  [{scope_desc}]")
    print(f"  Title  : {title}")
    if meta:
        print(f"  File   : {meta.get('file', '')}")
        if meta.get("page_range"):
            pr = meta["page_range"]
            print(f"  Pages  : {pr[0]}–{pr[1]}")
        if meta.get("chapter"):
            print(f"  Chapter: {meta['chapter']}")
        timing = meta.get("timing_seconds", {})
        print(
            f"  Time   : {timing.get('total', 0):.2f}s  "
            f"(load={timing.get('load', 0):.2f}s  "
            f"plan={timing.get('plan', 0):.2f}s  "
            f"work={timing.get('work', 0):.2f}s  "
            f"review={timing.get('review', 0):.2f}s  "
            f"write={timing.get('write', 0):.2f}s)"
        )
    print(sep)

    if section_summaries:
        print("\n── SECTION SUMMARIES ──────────────────────────────────────────────")
        for sec in section_summaries:
            print(f"\n  [{sec.get('section_id', '?')}] {sec.get('title', '')}")
            print(f"  {sec.get('summary', '').strip()}")

    print(f"\n{sep}")
    print("  FINAL SUMMARY")
    print(sep)
    print(final_summary)
    print()

    if meta.get("error"):
        print(f"⚠  Pipeline error: {meta['error']}\n")


if __name__ == "__main__":
    main()
