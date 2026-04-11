"""
agents/worker_agent.py — Summarises a single document section.

Each call is independent so that workers can run concurrently.

FIX — Blunt character hard-cap:
  The old code used `text[:8000]` (character slice) as a context-window guard.
  8 000 characters ≈ 2 000 tokens for plain English, but can be far fewer for
  code-heavy or non-Latin text, or far more for sparse/whitespace-heavy text.

  New design: use count_tokens() (already available in utils/chunker) to
  measure the actual token count, then truncate at the token boundary using
  tiktoken's encode/decode round-trip.  This ensures we never exceed the
  model's context regardless of language or document type.

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

import tiktoken

from config import Config
from utils.llm_client import call_llm
from utils.chunker import count_tokens
from utils.logger import get_logger

log = get_logger(__name__)

# Hard token limit for a single worker's input text.
# Leaves headroom for the prompt template + JSON response within max_tokens=4096.
_MAX_INPUT_TOKENS = 3000

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


# ─── Token-safe text truncation ───────────────────────────────────────────────

def _truncate_to_tokens(text: str, max_tokens: int, model: str) -> str:
    """
    Return *text* truncated to at most *max_tokens* tokens using tiktoken's
    encode/decode round-trip.  This is accurate regardless of language,
    character width, or whitespace density.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    token_ids = enc.encode(text)
    if len(token_ids) <= max_tokens:
        return text                         # already within limit

    truncated = enc.decode(token_ids[:max_tokens])
    log.debug(
        "Worker: text truncated from %d to %d tokens.",
        len(token_ids),
        max_tokens,
    )
    return truncated


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

    # Token-accurate truncation (fixes the blunt char-slice bug)
    actual_tokens = count_tokens(text, cfg.model)
    if actual_tokens > _MAX_INPUT_TOKENS:
        text = _truncate_to_tokens(text, _MAX_INPUT_TOKENS, cfg.model)
        log.info(
            "Worker[%d]: input truncated from %d → %d tokens.",
            section_id, actual_tokens, _MAX_INPUT_TOKENS,
        )

    log.info("Worker[%d]: summarising section '%s' …", section_id, title[:60])

    depth_instruction = _DEPTH_INSTRUCTIONS.get(cfg.summary_depth, _DEPTH_INSTRUCTIONS["medium"])
    mode_instruction  = _MODE_INSTRUCTIONS.get(cfg.output_mode,    _MODE_INSTRUCTIONS["paragraph"])

    prompt = _PROMPT_TEMPLATE.format(
        title=title,
        text=text,
        depth_instruction=depth_instruction,
        mode_instruction=mode_instruction,
        section_id=section_id,
    )

    try:
        result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)
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
