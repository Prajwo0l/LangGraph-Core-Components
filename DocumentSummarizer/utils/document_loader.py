"""
utils/document_loader.py — Load and clean PDF / TXT documents.

Supports:
  • PyMuPDF (fitz) for PDF – fast, handles most layouts
  • pdfplumber as fallback for complex PDFs
  • Plain-text (.txt) files

Targeting options (all optional — if none given the full document is loaded):
  • page_range  : (start, end) 1-based inclusive page numbers  → PDF only
  • chapter     : string like "Chapter 2" or "Introduction"   → PDF + TXT
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ─── Public entry point ──────────────────────────────────────────────────────

def load_document(
    file_path: str | Path,
    page_range: tuple[int, int] | None = None,   # (start, end) 1-based, inclusive
    chapter: str | None = None,                   # e.g. "Chapter 2" or "Introduction"
) -> str:
    """
    Load a PDF or TXT file and return cleaned plain text.

    Parameters
    ----------
    file_path  : Path to .pdf or .txt file.
    page_range : Only for PDF. Extract pages start..end (1-based, inclusive).
                 E.g. (3, 7) extracts pages 3, 4, 5, 6, 7.
    chapter    : Extract a named chapter / section from the text.
                 Works for both PDF and TXT.
                 Matched case-insensitively against heading lines.

    Raises
    ------
    FileNotFoundError  – file does not exist
    ValueError         – unsupported extension or invalid page range
    RuntimeError       – parsing failure or chapter not found
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    log.info("Loading document: %s (type=%s)", path.name, ext)

    if page_range:
        log.info("  → Page range: %d – %d", page_range[0], page_range[1])
    if chapter:
        log.info("  → Chapter filter: %r", chapter)

    # ── Load raw text ─────────────────────────────────────────────────────────
    if ext == ".pdf":
        raw = _load_pdf(path, page_range=page_range)
    elif ext == ".txt":
        if page_range:
            log.warning("--pages is only supported for PDF files. Ignoring for .txt")
        raw = _load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Use .pdf or .txt")

    if not raw.strip():
        raise RuntimeError(f"Document appears to be empty after parsing: {path}")

    cleaned = clean_text(raw)

    # ── Chapter extraction (works for both PDF and TXT) ───────────────────────
    if chapter:
        cleaned = _extract_chapter(cleaned, chapter)

    log.info("Document loaded – %d characters after cleaning.", len(cleaned))
    return cleaned


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _load_pdf(path: Path, page_range: tuple[int, int] | None = None) -> str:
    """Try PyMuPDF first; fall back to pdfplumber."""
    text = _load_pdf_pymupdf(path, page_range)
    if not text.strip():
        log.warning("PyMuPDF returned empty text – trying pdfplumber.")
        text = _load_pdf_pdfplumber(path, page_range)
    return text


def _load_pdf_pymupdf(path: Path, page_range: tuple[int, int] | None) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        total_pages = len(doc)

        if page_range:
            start, end = page_range
            # Validate
            if start < 1 or end < start or end > total_pages:
                raise ValueError(
                    f"Invalid page range ({start}-{end}) for a {total_pages}-page document."
                )
            page_indices = range(start - 1, end)   # fitz uses 0-based
            log.info("PDF has %d pages; extracting pages %d–%d.", total_pages, start, end)
        else:
            page_indices = range(total_pages)
            log.info("PDF has %d pages; extracting all.", total_pages)

        pages: list[str] = []
        for i in page_indices:
            pages.append(doc[i].get_text("text"))
        doc.close()
        return "\n".join(pages)

    except ImportError:
        log.warning("PyMuPDF (fitz) not installed – skipping.")
        return ""
    except ValueError:
        raise
    except Exception as exc:
        log.warning("PyMuPDF failed: %s", exc)
        return ""


def _load_pdf_pdfplumber(path: Path, page_range: tuple[int, int] | None) -> str:
    try:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)

            if page_range:
                start, end = page_range
                if start < 1 or end < start or end > total_pages:
                    raise ValueError(
                        f"Invalid page range ({start}-{end}) for a {total_pages}-page document."
                    )
                target_pages = pdf.pages[start - 1 : end]
            else:
                target_pages = pdf.pages

            for page in target_pages:
                text = page.extract_text() or ""
                pages.append(text)

        return "\n".join(pages)

    except ImportError:
        raise RuntimeError(
            "Neither PyMuPDF nor pdfplumber is installed. "
            "Run: pip install pymupdf pdfplumber"
        )
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"pdfplumber failed to parse {path}: {exc}") from exc


def _load_txt(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path} with any supported encoding.")


# ─── Chapter extraction ───────────────────────────────────────────────────────

# Matches common heading patterns:
#   Chapter 1 / Chapter One / CHAPTER 1
#   Section 3.2 / SECTION 3
#   Introduction / Conclusion / Abstract / References  (common standalone headings)
#   ## Markdown heading
_HEADING_RE = re.compile(
    r"(?mi)^(?:"
    r"#{1,6}\s+.+"                          # Markdown
    r"|(?:chapter|section|part)\s+[\w\d.]+"  # Chapter / Section / Part N
    r"|[A-Z][A-Za-z\s]{2,60}$"              # Title-case or ALL-CAPS standalone line
    r")"
)


def _extract_chapter(text: str, chapter: str) -> str:
    """
    Extract the text belonging to the named chapter / section.

    Strategy
    --------
    1. Find all heading positions in the text.
    2. Locate the first heading that contains *chapter* (case-insensitive).
    3. Return text from that heading up to (but not including) the next heading.

    Raises RuntimeError if no matching heading is found.
    """
    chapter_lower = chapter.strip().lower()

    # Find all headings with their start positions
    headings: list[tuple[int, str]] = []   # (start_pos, heading_text)
    for match in _HEADING_RE.finditer(text):
        headings.append((match.start(), match.group().strip()))

    if not headings:
        raise RuntimeError(
            f"No headings detected in the document. "
            f"Cannot extract chapter {chapter!r}."
        )

    # Find matching heading
    match_idx: int | None = None
    for i, (pos, heading_text) in enumerate(headings):
        if chapter_lower in heading_text.lower():
            match_idx = i
            break

    if match_idx is None:
        available = "\n  ".join(h for _, h in headings)
        raise RuntimeError(
            f"Chapter/section {chapter!r} not found.\n"
            f"Available headings:\n  {available}"
        )

    start_pos = headings[match_idx][0]
    matched_heading = headings[match_idx][1]

    # End is the start of the next heading (or end of text)
    if match_idx + 1 < len(headings):
        end_pos = headings[match_idx + 1][0]
    else:
        end_pos = len(text)

    extracted = text[start_pos:end_pos].strip()

    log.info(
        "Chapter extracted: %r  (%d characters)",
        matched_heading,
        len(extracted),
    )
    return extracted


# ─── Text cleaning ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise and clean raw extracted text:
      1. Unicode normalisation (NFC)
      2. Replace non-breaking / exotic whitespace
      3. Remove control characters (keep newlines/tabs)
      4. Collapse repeated blank lines → max 2
      5. Strip trailing whitespace per line
      6. Attempt to re-join broken sentences (heuristic)
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<![.!?])\n(?=[a-z])", " ", text)
    return text.strip()
