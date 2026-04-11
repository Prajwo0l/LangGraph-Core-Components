"""
agents/planner_agent.py — Splits the full document into logical sections.

FIX — Fundamental design flaw:
  The old implementation sent only the first 6 000 characters to the LLM but
  then asked it to return "text": "<full section text>" for ALL sections.
  The LLM cannot return text it has never seen — it either truncated or
  hallucinated content.

  New design (two clean responsibilities):
    1. LLM's job  → STRUCTURE ONLY: given the full text, return section
                    titles and their start/end character positions (boundaries).
    2. Our job    → TEXT SLICING: use those boundaries to cut the real text
                    ourselves.  The LLM never has to reproduce content.

  If the LLM boundary detection fails for any reason, we fall back to the
  raw chunks from the smart chunker (unchanged from before).

Output schema
─────────────
{
  "title": "<inferred document title>",
  "sections": [
    {"section_id": 0, "title": "Introduction",       "text": "..."},
    {"section_id": 1, "title": "Chapter 1: ...",     "text": "..."},
    ...
  ]
}
"""

import re
from typing import Any

from config import Config
from utils.llm_client import call_llm
from utils.chunker import chunk_document
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Prompt template ─────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a document structure analyst. "
    "Your ONLY job is to identify the title and section headings of a document. "
    "Do NOT reproduce or summarise any content — return headings only."
)

_PROMPT_TEMPLATE = """\
Read the document below and identify its structure.

Document:
\"\"\"
{text}
\"\"\"

Return a JSON object with:
  - "title": the document title (infer from the first heading or opening text)
  - "headings": an ordered list of every section heading you can find,
                exactly as they appear in the document text

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "title": "<document title>",
  "headings": [
    "<exact heading text as it appears>",
    ...
  ]
}}

Rules:
- Copy headings VERBATIM — do not paraphrase or reword.
- Include ALL headings from the entire document.
- If there are no clear headings, return an empty list: "headings": []
"""


# ─── Public function ─────────────────────────────────────────────────────────

def run_planner(full_text: str, cfg: Config) -> dict[str, Any]:
    """
    Split *full_text* into sections using a clean two-step process:
      1. Ask the LLM for structure (title + headings) — not for content.
      2. Slice the real text at those heading boundaries ourselves.

    Falls back to raw chunks from the smart chunker on any LLM failure.
    """
    log.info("Planner agent: analysing document structure …")

    # Step 1 — get raw chunks as fallback baseline
    raw_chunks = chunk_document(full_text, cfg)
    log.info("Planner: %d raw chunks from chunker.", len(raw_chunks))

    # Step 2 — ask LLM for STRUCTURE ONLY (headings + title)
    # Send the full text so the LLM can see all headings, not just the first N chars.
    # Respect the model's context window via the existing max_tokens / chunk limits.
    prompt = _PROMPT_TEMPLATE.format(text=full_text)

    doc_title = "Untitled Document"
    sections: list[dict] = []

    try:
        llm_result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)
        doc_title = llm_result.get("title", "Untitled Document")
        headings: list[str] = llm_result.get("headings", [])

        if headings:
            sections = _slice_by_headings(full_text, headings)
            log.info("Planner: sliced %d sections from LLM-detected headings.", len(sections))
        else:
            log.info("Planner: LLM found no headings — using chunker fallback.")

    except Exception as exc:
        log.warning("Planner LLM call failed (%s) – using chunker fallback.", exc)

    # Step 3 — fall back to raw chunks if LLM path produced nothing
    if not sections:
        sections = [
            {
                "section_id": chunk["index"],
                "title": f"Section {chunk['index'] + 1}",
                "text": chunk["text"],
            }
            for chunk in raw_chunks
        ]
        log.info("Planner: using %d chunk-based sections (fallback).", len(sections))

    return {"title": doc_title, "sections": sections}


# ─── Text slicing helper ──────────────────────────────────────────────────────

def _slice_by_headings(text: str, headings: list[str]) -> list[dict]:
    """
    Given the full document text and a list of heading strings returned by
    the LLM, find each heading's position in the text and slice out the
    content that belongs to it.

    Strategy
    --------
    - Search for each heading using re.search with re.MULTILINE so we match
      at the start of a line.
    - Sort found positions.
    - Slice from each heading's start to the next heading's start.
    - Sections whose heading cannot be found in the text are skipped.
    """
    found: list[tuple[int, str]] = []   # (start_pos, heading_text)

    for heading in headings:
        # Escape the heading for regex, then look for it at the start of any line
        pattern = r"(?m)^" + re.escape(heading.strip())
        m = re.search(pattern, text)
        if m:
            found.append((m.start(), heading.strip()))
        else:
            log.debug("Planner: heading not found in text – skipping: %r", heading[:60])

    if not found:
        return []

    # Sort by position (LLM may return headings out of order)
    found.sort(key=lambda x: x[0])

    sections: list[dict] = []
    for i, (start, heading) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append({
                "section_id": i,
                "title": heading,
                "text": section_text,
            })

    return sections
