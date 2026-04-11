"""
utils/chunker.py — Smart document chunking.

Strategy (priority order):
  1. Semantic boundaries – headings, chapter markers, paragraph breaks
  2. Token-based fallback – if a semantic chunk still exceeds max_chunk_tokens

FIX — _merge_small_chunks:
  The old implementation merged greedily across semantic boundaries.
  A small "Chapter 1 Conclusion" chunk would silently merge into
  "Chapter 2 Introduction" — destroying the section boundaries we worked
  hard to find in _semantic_split.

  New design: each chunk carries a `section_tag` (the first heading line
  found in its text, or a positional tag).  _merge_small_chunks only merges
  adjacent chunks that share the same section_tag.
"""

import re
from typing import Optional

import tiktoken

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

# Heading patterns used to detect semantic boundaries
_HEADING_RE = re.compile(
    r"(?m)^(?:"
    r"#{1,6}\s+"              # Markdown headings
    r"|Chapter\s+\d+"         # "Chapter N"
    r"|Section\s+\d+"         # "Section N"
    r"|PART\s+[IVXLC\d]+"     # "PART IV"
    r"|[A-Z][A-Z\s]{4,50}$"   # ALL-CAPS short line (≥5 chars)
    r")"
)


# ─── Tokenizer helper ─────────────────────────────────────────────────────────

def _get_encoder(model: str = "gpt-4o-mini") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = _get_encoder(model)
    return len(enc.encode(text))


# ─── Section tag helper ───────────────────────────────────────────────────────

def _extract_section_tag(text: str) -> str:
    """
    Return the first heading line found in *text* (used as a section identity
    tag so that _merge_small_chunks never crosses section boundaries).
    Falls back to the first non-empty line if no heading is found.
    """
    m = _HEADING_RE.search(text)
    if m:
        return m.group().strip()
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first_line[:80]          # cap tag length for readability


# ─── Public API ───────────────────────────────────────────────────────────────

def chunk_document(text: str, cfg: Config) -> list[dict]:
    """
    Split *text* into a list of chunk dicts:
        {"index": int, "text": str, "tokens": int, "section_tag": str}

    Uses semantic splitting first, then token-based splitting as fallback.
    Merging only happens within the same semantic section.
    """
    log.debug("Starting semantic chunking …")
    raw_sections = _semantic_split(text)
    log.debug("Semantic split produced %d sections.", len(raw_sections))

    # For each semantic section, split further if too large, keeping section_tag
    chunks: list[dict] = []
    for sec in raw_sections:
        tag = _extract_section_tag(sec)
        sub = _ensure_token_limit(sec, cfg)
        for chunk in sub:
            chunk["section_tag"] = tag
        chunks.extend(sub)

    # Merge tiny adjacent chunks — only within the same semantic section
    chunks = _merge_small_chunks(chunks, cfg)

    # Assign sequential indices
    for i, chunk in enumerate(chunks):
        chunk["index"] = i

    log.info("Chunking complete – %d chunks produced.", len(chunks))
    return chunks


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _semantic_split(text: str) -> list[str]:
    """Split on heading boundaries or double-newline paragraphs."""
    parts = _HEADING_RE.split(text)
    headings = _HEADING_RE.findall(text)

    sections: list[str] = []

    if len(headings) > 0:
        if parts[0].strip():
            sections.append(parts[0].strip())
        for heading, body in zip(headings, parts[1:]):
            combined = f"{heading}\n{body}".strip()
            if combined:
                sections.append(combined)
    else:
        sections = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    return sections if sections else [text]


def _ensure_token_limit(text: str, cfg: Config) -> list[dict]:
    """
    If *text* fits within max_chunk_tokens return it as-is.
    Otherwise recursively split by paragraphs, then by token boundary.
    """
    tokens = count_tokens(text, cfg.model)
    if tokens <= cfg.max_chunk_tokens:
        return [{"text": text, "tokens": tokens}]

    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paras) > 1:
        return _pack_paragraphs(paras, cfg)

    return _token_split(text, cfg)


def _pack_paragraphs(paras: list[str], cfg: Config) -> list[dict]:
    """Greedily pack paragraphs into chunks up to max_chunk_tokens."""
    chunks: list[dict] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paras:
        t = count_tokens(para, cfg.model)
        if current_tokens + t > cfg.max_chunk_tokens and current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append({"text": chunk_text, "tokens": current_tokens})
            current_parts = []
            current_tokens = 0

        if t > cfg.max_chunk_tokens:
            chunks.extend(_token_split(para, cfg))
        else:
            current_parts.append(para)
            current_tokens += t

    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append({"text": chunk_text, "tokens": current_tokens})

    return chunks


def _token_split(text: str, cfg: Config) -> list[dict]:
    """Hard token-boundary split with overlap."""
    enc = _get_encoder(cfg.model)
    token_ids = enc.encode(text)
    step = cfg.max_chunk_tokens - cfg.overlap_tokens
    chunks: list[dict] = []

    for start in range(0, len(token_ids), step):
        end = start + cfg.max_chunk_tokens
        slice_ids = token_ids[start:end]
        chunk_text = enc.decode(slice_ids)
        chunks.append({"text": chunk_text, "tokens": len(slice_ids)})

    return chunks


def _merge_small_chunks(chunks: list[dict], cfg: Config) -> list[dict]:
    """
    Merge consecutive small chunks — but ONLY within the same semantic section.

    A chunk is "small" when its token count is below min_chunk_tokens.
    Two chunks are in the same section when they share the same section_tag.
    Crossing a section boundary always flushes the buffer even if the result
    would fit within max_chunk_tokens.
    """
    merged: list[dict] = []
    buffer_text = ""
    buffer_tokens = 0
    buffer_tag = None

    for chunk in chunks:
        t = chunk["tokens"]
        tag = chunk.get("section_tag", "")

        # Flush buffer if we hit a new section boundary
        if buffer_tag is not None and tag != buffer_tag:
            if buffer_text:
                merged.append({
                    "text": buffer_text,
                    "tokens": buffer_tokens,
                    "section_tag": buffer_tag,
                })
            buffer_text = ""
            buffer_tokens = 0
            buffer_tag = None

        # Merge into buffer if it still fits within the same section
        if buffer_tokens + t <= cfg.max_chunk_tokens:
            buffer_text = (
                (buffer_text + "\n\n" + chunk["text"]).strip()
                if buffer_text else chunk["text"]
            )
            buffer_tokens += t
            buffer_tag = tag
        else:
            # Current buffer is full — flush it, start a new buffer
            if buffer_text:
                merged.append({
                    "text": buffer_text,
                    "tokens": buffer_tokens,
                    "section_tag": buffer_tag,
                })
            buffer_text = chunk["text"]
            buffer_tokens = t
            buffer_tag = tag

    if buffer_text:
        merged.append({
            "text": buffer_text,
            "tokens": buffer_tokens,
            "section_tag": buffer_tag,
        })

    return merged
