"""
config.py — Central configuration for the Document Summarization System.
All tunable parameters live here; nothing is hardcoded elsewhere.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # ── LLM ──────────────────────────────────────────────────────────────
    model: str = "gpt-4o-mini"          # any OpenAI-compatible model name
    temperature: float = 0.2
    max_tokens: int = 4096

    # ── Chunking ─────────────────────────────────────────────────────────
    max_chunk_tokens: int = 1500        # tokens before recursive split kicks in
    min_chunk_tokens: int = 100         # chunks smaller than this are merged
    overlap_tokens: int = 50            # token overlap between adjacent chunks

    # ── Summarization depth ───────────────────────────────────────────────
    # "short" | "medium" | "detailed"
    summary_depth: Literal["short", "medium", "detailed"] = "medium"

    # ── Output mode ───────────────────────────────────────────────────────
    # "bullet" | "paragraph"
    output_mode: Literal["bullet", "paragraph"] = "paragraph"

    # ── Parallelism ───────────────────────────────────────────────────────
    max_workers: int = 4                # concurrent worker agents

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"             # DEBUG | INFO | WARNING | ERROR
    log_file: str = "summarizer.log"


# Singleton used throughout the app
DEFAULT_CONFIG = Config()
