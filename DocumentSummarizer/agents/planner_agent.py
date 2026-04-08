"""
agents/planner_agent.py — Splits the full document into logical sections.

Output schema
─────────────
{
  "title": "<inferred document title>",
  "sections": [
    {
      "section_id": 0,
      "title": "Introduction",
      "text": "..."
    },
    ...
  ]
}
"""

import json
from typing import Any

from config import Config
from utils.llm_client import call_llm
from utils.chunker import chunk_document
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Prompt template ─────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a document analysis expert. "
    "Your job is to read the beginning of a document and identify its overall "
    "title and a structured list of logical sections."
)

_PROMPT_TEMPLATE = """\
Below is the beginning of a document (up to 3000 tokens).

Document excerpt:
\"\"\"
{excerpt}
\"\"\"

Respond ONLY with a valid JSON object (no markdown, no extra text) that has
this exact structure:
{{
  "title": "<document title or best guess>",
  "sections": [
    {{"section_id": 0, "title": "<section title>", "text": "<full section text>"}},
    ...
  ]
}}

Rules:
- Use the actual section headings found in the text as section titles.
- If no headings are found, create logical divisions (Introduction, Body, Conclusion, etc.).
- Preserve all original text in the "text" field without paraphrasing.
- Return ALL sections found; do not truncate.
"""


# ─── Public function ─────────────────────────────────────────────────────────

def run_planner(full_text: str, cfg: Config) -> dict[str, Any]:
    """
    Split *full_text* into sections.

    Strategy
    --------
    1. Use the chunker to create a list of raw chunks.
    2. Ask the LLM to identify section boundaries using the first ~3000 tokens
       as context for title / heading detection.
    3. Map each chunk back to a section dict.

    Returns a dict matching the schema above.
    """
    log.info("Planner agent: analysing document structure …")

    # Step 1 – get raw chunks from the smart chunker
    raw_chunks = chunk_document(full_text, cfg)
    log.info("Planner: %d raw chunks from chunker.", len(raw_chunks))

    # Step 2 – use LLM to infer title + section names from first excerpt
    excerpt = full_text[:6000]          # first ~6 000 chars for title detection
    prompt = _PROMPT_TEMPLATE.format(excerpt=excerpt)

    try:
        llm_result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)
        doc_title: str = llm_result.get("title", "Untitled Document")
        llm_sections: list[dict] = llm_result.get("sections", [])
    except Exception as exc:
        log.warning("Planner LLM call failed (%s) – falling back to chunk-based sections.", exc)
        doc_title = "Untitled Document"
        llm_sections = []

    # Step 3 – if the LLM gave us well-formed sections use them;
    #           otherwise fall back to the raw chunker output.
    if llm_sections and all("text" in s for s in llm_sections):
        sections = [
            {
                "section_id": i,
                "title": s.get("title", f"Section {i+1}"),
                "text": s.get("text", ""),
            }
            for i, s in enumerate(llm_sections)
            if s.get("text", "").strip()
        ]
        log.info("Planner: using %d LLM-detected sections.", len(sections))
    else:
        # Fall back: each chunk becomes a section
        sections = [
            {
                "section_id": chunk["index"],
                "title": f"Section {chunk['index'] + 1}",
                "text": chunk["text"],
            }
            for chunk in raw_chunks
        ]
        log.info("Planner: using %d chunk-based sections (LLM fallback).", len(sections))

    return {
        "title": doc_title,
        "sections": sections,
    }
