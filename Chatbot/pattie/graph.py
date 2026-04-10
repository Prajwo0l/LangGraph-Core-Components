# pattie/graph.py
# =============================================================================
# Graph assembly — the ONLY place that wires nodes and edges together.
#
# Reading this file is all you need to understand the full control flow.
# Every node is imported from its own module; no logic lives here.
#
#  START
#    │
#    ▼
#  intent_router  ──► chat_node ──┬──(no tool calls)──────────► ltm_update ──► END
#                                 │
#                                 ├──(read_file on .pdf)────────► pdf_ingest ──► chat_node
#                                 │
#                                 ├──(write/delete/multi-write)─► fs_hitl ──► END
#                                 │
#                                 └──(any other tool call)──────► tool_runner ──► chat_node
# =============================================================================
from __future__ import annotations

from langchain_core.messages import AIMessage

from langgraph.graph import END, START, StateGraph

from .config import CHATBOT_CONFIG_DEFAULTS, FS_HITL_TOOLS
from .nodes.intent_router import intent_router
from .nodes.chat          import chat_node
from .nodes.tool_runner   import tool_runner
from .nodes.pdf_ingest    import pdf_ingest_node, _find_pdf_read_call
from .nodes.fs_hitl       import fs_hitl_node, _find_hitl_call
from .nodes.ltm_update    import ltm_update_node
from .persistence         import checkpointer
from .state               import ChatState


# ── Edge routing function ─────────────────────────────────────────────────────

def _route_after_chat(state: ChatState) -> str:
    """
    Decide what happens after chat_node produces a response.

    Priority order (first match wins):
      1. No tool calls at all  → update LTM, then end the turn.
      2. read_file on a .pdf   → intercept and ingest the PDF.
      3. Destructive FS tool   → pause and wait for human approval.
      4. Any other tool call   → execute normally and loop back.

    INVARIANT: every branch MUST ensure that all tool_call_ids in the last
    AIMessage receive a paired ToolMessage before the graph ends.
    """
    last = state["messages"][-1]

    # No tool calls → conversation turn is complete
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return "ltm_update"

    # PDF intercept (check before HITL — read_file on .pdf is always auto)
    if _find_pdf_read_call(state["messages"]) is not None:
        return "pdf_ingest"

    # Filesystem HITL (write / delete need approval)
    if _find_hitl_call(state["messages"]) is not None:
        return "fs_hitl"

    # All other tool calls (search, calculator, stock price, expense, etc.)
    return "tool_runner"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the Pattie LangGraph.
    Call once at startup; the returned compiled graph is thread-safe.
    """
    g = StateGraph(ChatState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("intent_router", intent_router)
    g.add_node("chat_node",     chat_node)
    g.add_node("tool_runner",   tool_runner)
    g.add_node("pdf_ingest",    pdf_ingest_node)
    g.add_node("fs_hitl",       fs_hitl_node)
    g.add_node("ltm_update",    ltm_update_node)

    # ── Static edges ──────────────────────────────────────────────────────────
    g.add_edge(START,          "intent_router")
    g.add_edge("intent_router","chat_node")
    g.add_edge("ltm_update",   END)           # LTM updated → turn complete
    g.add_edge("tool_runner",  "chat_node")   # tool result → model sees it → continues
    g.add_edge("pdf_ingest",   "chat_node")   # PDF loaded  → model answers from RAG
    g.add_edge("fs_hitl",      END)           # paused      → frontend resumes after approval

    # ── Conditional edge out of chat_node ────────────────────────────────────
    g.add_conditional_edges(
        "chat_node",
        _route_after_chat,
        {
            "ltm_update":  "ltm_update",
            "pdf_ingest":  "pdf_ingest",
            "fs_hitl":     "fs_hitl",
            "tool_runner": "tool_runner",
        },
    )

    return g.compile(checkpointer=checkpointer)
