"""
agents/worker_agent.py — Summarises a single document section.

Each call is independent so that workers can run concurrently.

Output schema (per section)
───────────────────────────
{
  "section_id": 0,
  "title": "Introduction",
  "key_points": ["...", "..."],
  "summary": "..."
}
"""

from typing import Any

from config import Config
from utils.llm_client import call_llm
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Prompt templates ────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert summarization assistant. "
    "You produce accurate, concise, faithful summaries that preserve the "
    "meaning of the original text without hallucinating."
)

_DEPTH_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "Produce a SHORT summary (2–3 sentences) and at most 3 key points. "
        "Be extremely concise."
    ),
    "medium": (
        "Produce a MEDIUM-length summary (4–6 sentences) and 4–6 key points. "
        "Balance detail and brevity."
    ),
    "detailed": (
        "Produce a DETAILED summary (7–10 sentences) and 6–10 key points. "
        "Preserve important nuances and data."
    ),
}

_MODE_INSTRUCTIONS: dict[str, str] = {
    "bullet": "Write the summary as a set of bullet points.",
    "paragraph": "Write the summary as flowing prose paragraphs.",
}

_PROMPT_TEMPLATE = """\
Section title: {title}

Section text:
\"\"\"
{text}
\"\"\"

{depth_instruction}
{mode_instruction}

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{
  "section_id": {section_id},
  "title": "{title}",
  "key_points": ["<point 1>", "<point 2>", ...],
  "summary": "<summary text>"
}}
"""


# ─── Public function ─────────────────────────────────────────────────────────

def run_worker(section: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """
    Summarise a single *section* dict:
        {"section_id": int, "title": str, "text": str}

    Returns the output schema dict shown above.
    Falls back to a basic stub on LLM failure.
    """
    section_id: int = section["section_id"]
    title: str = section.get("title", f"Section {section_id + 1}")
    text: str = section.get("text", "")

    if not text.strip():
        log.warning("Worker[%d]: empty text – returning empty summary.", section_id)
        return {
            "section_id": section_id,
            "title": title,
            "key_points": [],
            "summary": "(Empty section)",
        }

    log.info("Worker[%d]: summarising section '%s' …", section_id, title[:60])

    depth_instruction = _DEPTH_INSTRUCTIONS.get(cfg.summary_depth, _DEPTH_INSTRUCTIONS["medium"])
    mode_instruction = _MODE_INSTRUCTIONS.get(cfg.output_mode, _MODE_INSTRUCTIONS["paragraph"])

    prompt = _PROMPT_TEMPLATE.format(
        title=title,
        text=text[:8000],               # hard cap to stay within context window
        depth_instruction=depth_instruction,
        mode_instruction=mode_instruction,
        section_id=section_id,
    )

    try:
        result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)
        # Ensure required keys exist
        result.setdefault("section_id", section_id)
        result.setdefault("title", title)
        result.setdefault("key_points", [])
        result.setdefault("summary", "")
        log.debug("Worker[%d]: done.", section_id)
        return result
    except Exception as exc:
        log.error("Worker[%d]: LLM call failed – %s", section_id, exc)
        return {
            "section_id": section_id,
            "title": title,
            "key_points": [],
            "summary": f"[Summarization failed for this section: {exc}]",
        }
