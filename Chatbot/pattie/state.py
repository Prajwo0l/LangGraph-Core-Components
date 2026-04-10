# pattie/state.py
# =============================================================================
# Defines the single shared state object that flows through the entire graph.
# Every node reads from and writes back to this TypedDict.
# =============================================================================
from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    # ── Core conversation ────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Thread metadata ──────────────────────────────────────────────────────
    title: str                  # Human-readable thread title shown in the sidebar

    # ── Routing ─────────────────────────────────────────────────────────────
    intent: str                 # One of: expense | filesystem | search | document | finance | general

    # ── Human-in-the-loop (HITL) ────────────────────────────────────────────
    # When a destructive filesystem tool is requested, the pending call is
    # stored here so the frontend can render an approval widget.
    # Structure: {'tool_name': str, 'args': dict, 'tool_call_id': str} | None
    fs_hitl_pending: Optional[dict]


# Sensible defaults used when creating a brand-new thread
DEFAULT_STATE: dict = {
    "title": "New Chat",
    "intent": "general",
    "fs_hitl_pending": None,
}
