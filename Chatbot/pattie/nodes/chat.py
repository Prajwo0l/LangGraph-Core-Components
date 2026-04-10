# pattie/nodes/chat.py
# =============================================================================
# Node: chat_node
#
# Responsibility: generate the assistant's response using the LLM bound to
# the tools relevant for the current intent. Also injects STM + LTM memory
# context into the system prompt.
#
# This is the only node that calls an LLM that produces a reply to the user.
# =============================================================================
from __future__ import annotations

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from ..memory import apply_stm, build_memory_context
from ..rag import get_active_thread
from ..state import ChatState
from ..tools.registry import llm_by_intent

# ── Per-intent system-prompt hints ───────────────────────────────────────────
# Describes what the LLM should focus on and which tools are available.
_INTENT_HINTS: dict[str, str] = {
    "expense": (
        "You are helping the user track expenses, budgets, and income. "
        "Use expense tools to add, list, edit, or summarize."
    ),
    "filesystem": (
        "You are helping the user manage files in their Downloads folder.\n"
        "Available tools:\n"
        "  • list_files          — browse the directory\n"
        "  • read_file           — open a single text file\n"
        "  • read_multiple_files — open several files at once (pass a list of paths)\n"
        "  • write_file          — create or overwrite a single file\n"
        "  • write_multiple_files — create or overwrite multiple files in one call\n"
        "                          (pass a dict: {filename: content, ...})\n"
        "  • delete_file / delete_folder — remove files or directories\n"
        "For .pdf files, call read_file with just the filename — PDF ingestion is automatic.\n"
        "Use only the filename relative to Downloads, never the full path.\n"
        "Prefer batch tools (read_multiple_files, write_multiple_files) when the user "
        "asks to read or create several files at once — it is more efficient."
    ),
    "search": (
        "You are helping the user find information. "
        "Use the search tool to look things up online."
    ),
    "document": (
        "A PDF document has been loaded into memory. "
        "Use rag_tool with the user's question to retrieve relevant passages and answer them."
    ),
    "finance": (
        "You are helping with financial data. "
        "Use get_stock_price or calculator as needed."
    ),
    "general": (
        "You are having a friendly conversation. "
        "No tools needed unless the user explicitly asks."
    ),
}


def chat_node(state: ChatState) -> dict:
    """
    Main LLM node.

    Steps:
      1. Compress old messages with STM sliding-window summary.
      2. Retrieve relevant LTM facts for the current query.
      3. Build a system message with intent hint + memory context.
      4. Call the intent-specific LLM binding.
    """
    today     = date.today().strftime("%Y-%m-%d")
    intent    = state.get("intent", "general")
    thread_id = get_active_thread()

    # ── 1. STM: compress old messages, keep last N verbatim ──────────────────
    stm_messages = apply_stm(thread_id, state["messages"])

    # ── 2. LTM: semantic retrieval based on current query ────────────────────
    last_human_content = next(
        (m.content for m in reversed(stm_messages) if isinstance(m, HumanMessage)),
        "",
    )
    memory_context = build_memory_context(thread_id, stm_messages, last_human_content)

    # ── 3. System message ─────────────────────────────────────────────────────
    memory_section = f"\n\n{memory_context}" if memory_context else ""
    system = SystemMessage(content=(
        f"You are Pattie, a helpful personal AI assistant.\n"
        f"Today's date is {today}.\n"
        f"When adding expenses, use {today} if the user does not specify a date.\n"
        f"Always pass all required arguments when calling tools.\n"
        f"Current task: {_INTENT_HINTS.get(intent, '')}"
        f"{memory_section}"
    ))

    # ── 4. LLM call ───────────────────────────────────────────────────────────
    active_llm = llm_by_intent.get(intent)
    response   = active_llm.invoke([system] + stm_messages)
    return {"messages": [response]}
