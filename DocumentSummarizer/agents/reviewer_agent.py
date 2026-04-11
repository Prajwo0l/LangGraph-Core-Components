"""
agents/reviewer_agent.py — Reviews and refines all section summaries.

Automatically batches large payloads that exceed REVIEWER_MAX_INPUT_TOKENS.
"""

import json
from typing import Any

from config import Config
from utils.llm_client import call_llm
from utils.chunker import count_tokens
from utils.logger import get_logger

log = get_logger(__name__)

REVIEWER_MAX_INPUT_TOKENS = 6000

_SYSTEM = (
    "You are a meticulous editorial reviewer. "
    "You review a set of section summaries and ensure they are consistent, "
    "non-redundant, and logically coherent. "
    "IMPORTANT: Your response must be valid JSON. "
    "Never use raw newlines or unescaped quotes inside JSON string values. "
    "Use \\n for newlines and \\\" for quotes inside strings."
)

_REVIEW_PROMPT = """\
You have section summaries from a document titled "{doc_title}".

Summaries (JSON):
{summaries_json}

Your tasks:
1. Remove or merge REDUNDANT key points that appear in more than one section.
2. Fix INCONSISTENT terminology across sections.
3. Identify and note any LOGICAL GAPS or contradictions.
4. Keep original meaning intact — do NOT add new information.

CRITICAL JSON RULES:
- Respond ONLY with the JSON object below — no markdown, no explanation.
- All string values must have special characters escaped: use \\n for newlines, \\" for quotes.
- Do not include raw line breaks inside any string value.

{{
  "reviewed_sections": [
    {{
      "section_id": <int>,
      "title": "<title>",
      "key_points": ["<refined point>", ...],
      "summary": "<refined summary — no raw newlines>",
      "reviewer_notes": "<optional note>"
    }},
    ...
  ],
  "global_notes": "<overall observations>"
}}
"""

_MERGE_PROMPT = """\
You are merging multiple review passes of a document titled "{doc_title}" into one.

All reviewed sections (JSON):
{all_reviewed_json}

Produce a single unified list. Remove cross-batch terminology inconsistencies.

CRITICAL JSON RULES:
- Respond ONLY with the JSON object below — no markdown, no explanation.
- All string values must have special characters escaped: use \\n for newlines, \\" for quotes.
- Do not include raw line breaks inside any string value.

{{
  "reviewed_sections": [
    {{
      "section_id": <int>,
      "title": "<title>",
      "key_points": ["..."],
      "summary": "<no raw newlines>",
      "reviewer_notes": "..."
    }},
    ...
  ],
  "global_notes": "<unified observations>"
}}
"""


def run_reviewer(
    doc_title: str,
    section_summaries: list[dict[str, Any]],
    cfg: Config,
) -> dict[str, Any]:
    """
    Review section_summaries with automatic batching when payload exceeds
    REVIEWER_MAX_INPUT_TOKENS.
    """
    if not section_summaries:
        log.warning("Reviewer: no sections to review.")
        return {"reviewed_sections": [], "global_notes": "No sections provided."}

    log.info("Reviewer agent: processing %d sections …", len(section_summaries))

    compact = _to_compact(section_summaries)
    payload_tokens = count_tokens(json.dumps(compact), cfg.model)
    log.info("Reviewer: payload = %d tokens (limit=%d).", payload_tokens, REVIEWER_MAX_INPUT_TOKENS)

    try:
        if payload_tokens <= REVIEWER_MAX_INPUT_TOKENS:
            result = _review_batch(doc_title, compact, cfg)
            log.info("Reviewer: single-pass done, %d sections.", len(result["reviewed_sections"]))
            return result
        else:
            batches = _split_into_batches(compact, cfg)
            log.info("Reviewer: split into %d batches.", len(batches))

            all_reviewed: list[dict] = []
            combined_notes: list[str] = []

            for i, batch in enumerate(batches):
                log.info("Reviewer: processing batch %d/%d …", i + 1, len(batches))
                batch_result = _review_batch(doc_title, batch, cfg)
                all_reviewed.extend(batch_result.get("reviewed_sections", []))
                note = batch_result.get("global_notes", "")
                if note:
                    combined_notes.append(note)

            if len(batches) > 1:
                log.info("Reviewer: running merge pass …")
                result = _merge_reviewed(doc_title, all_reviewed, combined_notes, cfg)
            else:
                result = {
                    "reviewed_sections": all_reviewed,
                    "global_notes": " ".join(combined_notes),
                }

            log.info("Reviewer: batched review done, %d sections.", len(result["reviewed_sections"]))
            return result

    except Exception as exc:
        log.error("Reviewer: failed – %s. Returning original summaries.", exc)
        fallback = [{**s, "reviewer_notes": f"[Review failed: {exc}]"} for s in compact]
        return {"reviewed_sections": fallback, "global_notes": f"Review failed: {exc}"}


def _to_compact(sections: list[dict]) -> list[dict]:
    return [
        {
            "section_id": s["section_id"],
            "title": s.get("title", ""),
            "key_points": s.get("key_points", []),
            "summary": s.get("summary", ""),
        }
        for s in sections
    ]


def _review_batch(doc_title: str, batch: list[dict], cfg: Config) -> dict[str, Any]:
    summaries_json = json.dumps(batch, indent=2, ensure_ascii=False)
    prompt = _REVIEW_PROMPT.format(doc_title=doc_title, summaries_json=summaries_json)

    result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)

    validated: list[dict] = []
    for sec in result.get("reviewed_sections", []):
        validated.append({
            "section_id": sec.get("section_id", 0),
            "title": sec.get("title", ""),
            "key_points": sec.get("key_points", []),
            "summary": sec.get("summary", ""),
            "reviewer_notes": sec.get("reviewer_notes", ""),
        })

    return {
        "reviewed_sections": validated,
        "global_notes": result.get("global_notes", ""),
    }


def _split_into_batches(sections: list[dict], cfg: Config) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for sec in sections:
        sec_tokens = count_tokens(json.dumps(sec), cfg.model)
        if current_tokens + sec_tokens > REVIEWER_MAX_INPUT_TOKENS and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(sec)
        current_tokens += sec_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _merge_reviewed(
    doc_title: str,
    all_reviewed: list[dict],
    notes: list[str],
    cfg: Config,
) -> dict[str, Any]:
    merge_tokens = count_tokens(json.dumps(all_reviewed), cfg.model)
    if merge_tokens > REVIEWER_MAX_INPUT_TOKENS * 2:
        log.warning(
            "Reviewer: merge payload (%d tokens) too large — skipping LLM merge.",
            merge_tokens,
        )
        all_reviewed.sort(key=lambda x: x.get("section_id", 0))
        return {
            "reviewed_sections": all_reviewed,
            "global_notes": " | ".join(notes),
        }

    all_reviewed_json = json.dumps(all_reviewed, indent=2, ensure_ascii=False)
    prompt = _MERGE_PROMPT.format(
        doc_title=doc_title,
        all_reviewed_json=all_reviewed_json,
    )

    try:
        result = call_llm(prompt, cfg, system_prompt=_SYSTEM, expect_json=True)
        merged: list[dict] = []
        for sec in result.get("reviewed_sections", []):
            merged.append({
                "section_id": sec.get("section_id", 0),
                "title": sec.get("title", ""),
                "key_points": sec.get("key_points", []),
                "summary": sec.get("summary", ""),
                "reviewer_notes": sec.get("reviewer_notes", ""),
            })
        merged.sort(key=lambda x: x["section_id"])
        return {
            "reviewed_sections": merged,
            "global_notes": result.get("global_notes", " | ".join(notes)),
        }
    except Exception as exc:
        log.warning("Reviewer: merge LLM call failed (%s) — returning sorted batches.", exc)
        all_reviewed.sort(key=lambda x: x.get("section_id", 0))
        return {
            "reviewed_sections": all_reviewed,
            "global_notes": " | ".join(notes),
        }
