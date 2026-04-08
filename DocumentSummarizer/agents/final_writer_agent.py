"""
agents/final_writer_agent.py — Produces the final, polished document summary.

Takes the reviewed section summaries and writes a clean, coherent, well-
structured final summary in the user's chosen output mode.

Output schema
─────────────
{
  "title": "<document title>",
  "section_summaries": [
    {"section_id": 0, "title": "...", "summary": "..."},
    ...
  ],
  "final_summary": "<full combined summary>"
}
"""

import json
from typing import Any

from config import Config
from utils.llm_client import call_llm
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Prompt templates ────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a world-class technical writer. "
    "You synthesize multiple section summaries into a single, fluent, "
    "well-structured final summary that reads as a standalone document."
)

_MODE_INSTRUCTIONS: dict[str, str] = {
    "bullet": (
        "Write the final summary as structured bullet points grouped by section. "
        "Each section should have a heading followed by its key points."
    ),
    "paragraph": (
        "Write the final summary as flowing prose. "
        "Blend the section summaries into a coherent narrative with smooth transitions. "
        "Use clear paragraph breaks between major topics."
    ),
}

_DEPTH_LENGTH: dict[str, str] = {
    "short": "Keep the final summary under 200 words.",
    "medium": "Aim for 300–500 words.",
    "detailed": "Aim for 600–1000 words; preserve important details.",
}

_PROMPT_TEMPLATE = """\
Document title: {doc_title}

Reviewed section summaries (JSON):
{reviewed_json}

{mode_instruction}
{depth_instruction}

Ensure:
- Logical flow from beginning to end
- No redundancy between sections
- Consistent terminology throughout
- Clarity and readability

Respond ONLY with a valid JSON object (no markdown):
{{
  "title": "{doc_title}",
  "section_summaries": [
    {{"section_id": <int>, "title": "<title>", "summary": "<section summary>"}},
    ...
  ],
  "final_summary": "<complete final summary>"
}}
"""


# ─── Public function ─────────────────────────────────────────────────────────

def run_final_writer(
    doc_title: str,
    reviewed_sections: list[dict[str, Any]],
    cfg: Config,
) -> dict[str, Any]:
    """
    Generate the final summary document.

    Returns the output schema dict shown above.
    """
    if not reviewed_sections:
        log.warning("Final writer: no reviewed sections to synthesize.")
        return {
            "title": doc_title,
            "section_summaries": [],
            "final_summary": "(No content to summarize.)",
        }

    log.info("Final writer: synthesizing %d reviewed sections …", len(reviewed_sections))

    compact = [
        {
            "section_id": s["section_id"],
            "title": s.get("title", ""),
            "key_points": s.get("key_points", []),
            "summary": s.get("summary", ""),
        }
        for s in reviewed_sections
    ]
    reviewed_json = json.dumps(compact, indent=2, ensure_ascii=False)

    mode_instruction = _MODE_INSTRUCTIONS.get(cfg.output_mode, _MODE_INSTRUCTIONS["paragraph"])
    depth_instruction = _DEPTH_LENGTH.get(cfg.summary_depth, _DEPTH_LENGTH["medium"])

    prompt = _PROMPT_TEMPLATE.format(
        doc_title=doc_title,
        reviewed_json=reviewed_json,
        mode_instruction=mode_instruction,
        depth_instruction=depth_instruction,
    )

    try:
        result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)
        result.setdefault("title", doc_title)
        result.setdefault("section_summaries", compact)
        result.setdefault("final_summary", "")
        log.info("Final writer: done.")
        return result
    except Exception as exc:
        log.error("Final writer: LLM call failed – %s. Assembling fallback output.", exc)
        fallback_summary = "\n\n".join(
            f"**{s.get('title', '')}**\n{s.get('summary', '')}" for s in compact
        )
        return {
            "title": doc_title,
            "section_summaries": compact,
            "final_summary": fallback_summary,
        }
