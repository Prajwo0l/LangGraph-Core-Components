"""
utils/llm_client.py — Thin wrapper around the OpenAI chat-completions API.

Loads OPENAI_API_KEY automatically from:
  1. DocumentSummarizer/.env   (local)
  2. LangGraph-Core-Components/.env  (parent — shared project .env)

All LLM calls in this project go through call_llm() so that:
  • retry logic lives in one place
  • model / temperature / max_tokens are taken from Config
  • JSON parsing with aggressive repair lives here

FIX — Invalid JSON from large responses
────────────────────────────────────────
For documents with many sections the reviewer / writer LLM responses can be
22 000+ characters.  The LLM occasionally embeds raw newlines, unescaped
double-quotes, or control characters inside JSON string values, which breaks
json.loads().

New _parse_json() strategy (4 progressive attempts):
  1. Direct json.loads on the stripped text                        (fast path)
  2. Strip markdown fences, try again
  3. Extract first { … } block, try again
  4. Run _repair_json() which fixes the most common LLM mistakes,
     then try json.loads one final time

_repair_json() handles:
  • Raw newlines / tabs inside string values  → \\n / \\t
  • Trailing commas before } or ]
  • Single-quoted strings → double-quoted  (rare but happens)
  • Control characters (\\x00–\\x1f except \\n\\r\\t)
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

# ── Load .env ─────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent.parent
_LOCAL_ENV = _THIS_DIR / ".env"
_PARENT_ENV = _THIS_DIR.parent / ".env"

if _LOCAL_ENV.exists():
    load_dotenv(_LOCAL_ENV, override=False)
elif _PARENT_ENV.exists():
    load_dotenv(_PARENT_ENV, override=False)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ─── Public API ───────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    cfg: Config,
    system_prompt: str = "You are a helpful assistant.",
    expect_json: bool = True,
    retries: int = 3,
    backoff: float = 2.0,
) -> dict[str, Any] | str:
    """
    Send *prompt* to the configured LLM and return the response.

    Parameters
    ----------
    prompt        : User-turn message.
    cfg           : Config instance (model, temperature, max_tokens).
    system_prompt : System-turn message.
    expect_json   : If True, parse response as JSON and return a dict.
                    If False, return raw string.
    retries       : Number of retry attempts on transient errors.
    backoff       : Initial back-off in seconds (doubles on each retry).

    Raises
    ------
    RuntimeError  – if all retries are exhausted or JSON cannot be repaired.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    last_error: Exception | None = None
    delay = backoff

    for attempt in range(1, retries + 1):
        try:
            log.debug("LLM call (attempt %d/%d) model=%s", attempt, retries, cfg.model)
            t0 = time.perf_counter()

            response = _get_client().chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

            elapsed = time.perf_counter() - t0
            log.debug("LLM responded in %.2fs", elapsed)

            content = response.choices[0].message.content or ""

            if not expect_json:
                return content

            return _parse_json(content)

        except RateLimitError as exc:
            log.warning("Rate limit hit – waiting %.1fs …", delay)
            last_error = exc
        except APITimeoutError as exc:
            log.warning("API timeout – waiting %.1fs …", delay)
            last_error = exc
        except APIError as exc:
            log.warning("API error: %s – waiting %.1fs …", exc, delay)
            last_error = exc
        except (json.JSONDecodeError, ValueError) as exc:
            # JSON parse error — raise immediately so reviewer can catch and fall back
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

        time.sleep(delay)
        delay *= 2

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


# ─── JSON parsing with progressive repair ─────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any]:
    """
    Parse JSON from an LLM response, applying progressive repair strategies.

    Attempt order:
      1. Direct json.loads on stripped text
      2. Strip markdown fences, try again
      3. Extract first { … } block, try again
      4. Run _repair_json(), try again
      5. Raise ValueError with a clear message
    """
    stripped = text.strip()

    # ── Attempt 1: direct parse ───────────────────────────────────────────────
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: strip markdown fences ─────────────────────────────────────
    fence_stripped = _strip_fences(stripped)
    if fence_stripped != stripped:
        try:
            return json.loads(fence_stripped)
        except json.JSONDecodeError:
            pass
    else:
        fence_stripped = stripped

    # ── Attempt 3: extract first { … } block ─────────────────────────────────
    extracted = _extract_json_block(fence_stripped)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # ── Attempt 4: repair + parse ─────────────────────────────────────────────
    to_repair = extracted or fence_stripped
    repaired = _repair_json(to_repair)
    try:
        result = json.loads(repaired)
        log.debug("_parse_json: repaired JSON parsed successfully.")
        return result
    except json.JSONDecodeError as exc:
        log.warning("_parse_json: all repair attempts failed. Error: %s", exc)
        raise ValueError(
            f"Could not parse LLM JSON after repair attempts.\n"
            f"Parse error: {exc}\n"
            f"First 400 chars of raw text:\n{text[:400]}"
        ) from exc


def _strip_fences(text: str) -> str:
    """Remove ``` or ```json ... ``` markdown fences."""
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json) and last line if it's ```
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[1:end]).strip()
    return text


def _extract_json_block(text: str) -> str:
    """
    Extract the outermost { … } block from text.
    Uses a simple brace-counter so it handles nested objects correctly.
    """
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Brace never closed — return everything from start
    return text[start:]


def _repair_json(text: str) -> str:
    """
    Apply heuristic fixes for the most common LLM JSON mistakes:

    1. Trailing commas before } or ]
       e.g.  {"a": 1,}  →  {"a": 1}
    2. Raw (unescaped) newlines/tabs inside string values
       This is the most common cause of "Expecting ',' delimiter" errors
       in large responses.
    3. Control characters inside strings (\\x00–\\x1f except \\n\\r\\t)
    4. Single-quoted strings → double-quoted (rare)
    """
    # ── 1. Fix trailing commas ────────────────────────────────────────────────
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # ── 2 & 3. Fix raw newlines/tabs/control chars inside string values ───────
    # Walk character by character; inside a JSON string, replace bare
    # newlines/tabs/control chars with their escape sequences.
    text = _fix_string_internals(text)

    # ── 4. Single-quoted strings (very rare but happens) ─────────────────────
    # Only attempt if standard parse still fails — this is a blunt heuristic
    # and can break legitimate apostrophes, so we only do it as a last resort.
    # (We leave this out for now; if needed it can be added.)

    return text


def _fix_string_internals(text: str) -> str:
    """
    Scan the JSON text and, while inside a string literal, replace:
      - bare \\n (0x0A) → \\\\n
      - bare \\r (0x0D) → \\\\r
      - bare \\t (0x09) → \\\\t
      - other control chars (0x00–0x1f) → removed

    Uses a simple state machine; correctly handles \\\" escapes so we
    don't accidentally toggle in_string on escaped quotes.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if in_string:
            if ch == "\\":
                # Escape sequence — copy both chars verbatim
                out.append(ch)
                i += 1
                if i < n:
                    out.append(text[i])
                i += 1
                continue
            elif ch == '"':
                # End of string
                in_string = False
                out.append(ch)
                i += 1
                continue
            elif ch == "\n":
                out.append("\\n")
                i += 1
                continue
            elif ch == "\r":
                out.append("\\r")
                i += 1
                continue
            elif ch == "\t":
                out.append("\\t")
                i += 1
                continue
            elif ord(ch) < 0x20:
                # Other control character — drop it
                i += 1
                continue
        else:
            if ch == '"':
                in_string = True

        out.append(ch)
        i += 1

    return "".join(out)
