"""
agents/reviewer_agent.py — Reviews and refines all section summaries.

Responsibilities:
  • Detect and remove redundant key points across sections
  • Fix logical inconsistencies / contradictions
  • Ensure consistent terminology throughout
  • Flag gaps (sections that seem under-summarised)

Output schema
─────────────
{
  "reviewed_sections": [
    {
      "section_id": 0,
      "title": "Introduction",
      "key_points": ["...", "..."],
      "summary": "...",
      "reviewer_notes": "..."   ← optional notes from reviewer
    },
    ...
  ],
  "global_notes": "..."         ← overall observations
}
"""

from typing import Any
import json

from config import Config
from utils.llm_client import call_llm
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Prompt template ─────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a meticulous editorial reviewer. "
    "You review a set of section summaries produced by different workers "
    "and ensure they are consistent, non-redundant, and logically coherent."
)

_PROMPT_TEMPLATE = """\
You have the following section-by-section summaries of a document titled
"{doc_title}".

Summaries (JSON array):
{summaries_json}

Your tasks:
1. Remove or merge REDUNDANT key points that appear in more than one section.
2. Fix INCONSISTENT terminology (e.g., if Section 1 calls something "model" and
   Section 3 calls the same thing "algorithm", unify the term).
3. Identify and note any LOGICAL GAPS or contradictions between sections.
4. Keep the original meaning intact – do NOT add new information.

Respond ONLY with a valid JSON object (no markdown):
{{
  "reviewed_sections": [
    {{
      "section_id": <int>,
      "title": "<title>",
      "key_points": ["<refined point>", ...],
      "summary": "<refined summary>",
      "reviewer_notes": "<optional short note about changes made>"
    }},
    ...
  ],
  "global_notes": "<overall observations about consistency or gaps>"
}}
"""


# ─── Public function ─────────────────────────────────────────────────────────

def run_reviewer(
    doc_title: str,
    section_summaries: list[dict[str, Any]],
    cfg: Config,
) -> dict[str, Any]:
    """
    Review *section_summaries* and return a refined version.

    Falls back gracefully: if the LLM call fails, the original summaries are
    returned wrapped in the expected schema.
    """
    if not section_summaries:
        log.warning("Reviewer: no sections to review.")
        return {"reviewed_sections": [], "global_notes": "No sections provided."}

    log.info("Reviewer agent: processing %d sections …", len(section_summaries))

    # Build a compact JSON representation of the summaries
    compact = [
        {
            "section_id": s["section_id"],
            "title": s.get("title", ""),
            "key_points": s.get("key_points", []),
            "summary": s.get("summary", ""),
        }
        for s in section_summaries
    ]
    summaries_json = json.dumps(compact, indent=2, ensure_ascii=False)

    prompt = _PROMPT_TEMPLATE.format(
        doc_title=doc_title,
        summaries_json=summaries_json,
    )

    try:
        result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)
        reviewed = result.get("reviewed_sections", [])
        global_notes = result.get("global_notes", "")

        # Validate and fill defaults
        validated: list[dict] = []
        for sec in reviewed:
            validated.append({
                "section_id": sec.get("section_id", 0),
                "title": sec.get("title", ""),
                "key_points": sec.get("key_points", []),
                "summary": sec.get("summary", ""),
                "reviewer_notes": sec.get("reviewer_notes", ""),
            })

        log.info("Reviewer: done. %d sections refined.", len(validated))
        return {"reviewed_sections": validated, "global_notes": global_notes}

    except Exception as exc:
        log.error("Reviewer: LLM call failed – %s. Returning original summaries.", exc)
        fallback = [
            {**s, "reviewer_notes": f"[Review failed: {exc}]"}
            for s in compact
        ]
        return {"reviewed_sections": fallback, "global_notes": f"Review failed: {exc}"}
