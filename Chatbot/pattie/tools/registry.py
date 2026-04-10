# pattie/tools/registry.py
# =============================================================================
# Single source of truth for:
#   • Which tools exist and what they are called
#   • How tools are grouped by intent
#   • Which LLM binding to use for each intent
#
# Nothing in the graph creates tool bindings — it always reads from here.
# To add a new intent or tool: edit TOOL_GROUPS, everything else updates
# automatically.
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, List

from ..models import llm
from .builtin import search_tool, calculator, get_stock_price, rag_tool
from .mcp_loader import load_mcp_tools

# ── Load MCP tools once at import time ───────────────────────────────────────
mcp_tools: List[Any] = load_mcp_tools()

# ── Named lookup: tool_name → tool object ────────────────────────────────────
_mcp_by_name: Dict[str, Any] = {t.name: t for t in mcp_tools}

# ── Expense tools ─────────────────────────────────────────────────────────────
_EXPENSE_TOOL_NAMES = {
    "add_expense", "add_to_calendar", "list_expenses", "summarize",
    "set_budget", "list_budgets", "check_budget_alerts", "delete_budget",
    "add_credit", "list_credits", "edit_credit", "delete_credit",
    "edit_expense", "delete_expense", "monthly_overview",
}

# ── Filesystem tools ──────────────────────────────────────────────────────────
# Includes both the original single-file tools AND the new multi-file tools.
_FILESYSTEM_TOOL_NAMES = {
    "list_files",
    "read_file",
    "read_multiple_files",   # ← NEW: read several files in one call
    "write_file",
    "write_multiple_files",  # ← NEW: write several files in one call
    "delete_file",
    "delete_folder",
}

_expense_tools    = [t for t in mcp_tools if t.name in _EXPENSE_TOOL_NAMES]
_filesystem_tools = [t for t in mcp_tools if t.name in _FILESYSTEM_TOOL_NAMES]

# ── Intent → tool list ────────────────────────────────────────────────────────
# Adding a new intent: add an entry here and a matching hint in intent_router.py
TOOL_GROUPS: Dict[str, List[Any]] = {
    "expense":    _expense_tools,
    "filesystem": _filesystem_tools,
    "search":     [search_tool],
    "document":   [rag_tool],
    "finance":    [get_stock_price, calculator],
    "general":    [],
}

# ── All tools in one flat list (used by the full-access LLM binding) ──────────
tools: List[Any] = [search_tool, calculator, get_stock_price, rag_tool] + mcp_tools

# ── Per-intent LLM bindings ──────────────────────────────────────────────────
# Each intent gets an LLM bound to only the tools it needs — this keeps the
# context window lean and prevents the model from reaching for irrelevant tools.
llm_by_intent: Dict[str, Any] = {
    intent: llm.bind_tools(tool_list) if tool_list else llm
    for intent, tool_list in TOOL_GROUPS.items()
}

# ── Named lookup: all tools ───────────────────────────────────────────────────
tools_by_name: Dict[str, Any] = {t.name: t for t in tools}
