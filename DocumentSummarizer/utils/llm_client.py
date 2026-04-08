"""
utils/llm_client.py — Thin wrapper around the OpenAI chat-completions API.

Loads OPENAI_API_KEY automatically from:
  1. DocumentSummarizer/.env   (local)
  2. LangGraph-Core-Components/.env  (parent — shared project .env)

All LLM calls in this project go through call_llm() so that:
  • retry logic lives in one place
  • model / temperature / max_tokens are taken from Config
  • JSON parsing errors are surfaced clearly
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

# ── Load .env (local first, then parent folder) ───────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent.parent   # DocumentSummarizer/
_LOCAL_ENV = _THIS_DIR / ".env"
_PARENT_ENV = _THIS_DIR.parent / ".env"

if _LOCAL_ENV.exists():
    load_dotenv(_LOCAL_ENV, override=False)
elif _PARENT_ENV.exists():
    load_dotenv(_PARENT_ENV, override=False)

# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()          # reads OPENAI_API_KEY from env
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
    RuntimeError  – if all retries are exhausted or JSON is malformed.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
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
            # JSON parse error – no point retrying the same prompt
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

        time.sleep(delay)
        delay *= 2

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any]:
    """
    Extract and parse the first JSON object/array from *text*.
    Handles responses wrapped in ```json … ``` fences.
    """
    # Strip markdown code fences
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        stripped = inner.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(stripped[start:end])
        raise ValueError(f"No valid JSON found in LLM output:\n{text[:300]}")
