# pattie/nodes/fs_hitl.py
# =============================================================================
# Node: fs_hitl  (Filesystem Human-In-The-Loop)
#
# Responsibility: intercept destructive filesystem tool calls
# (write_file, write_multiple_files, delete_file, delete_folder)
# and pause the graph, waiting for the user to approve or reject.
#
# Flow:
#   1. Detect a HITL-required tool call in the last AIMessage.
#   2. Store the pending call in state['fs_hitl_pending'] (frontend reads this).
#   3. Return a placeholder ToolMessage (satisfies tool_call_id pairing).
#   4. The graph ends here — frontend renders the approval widget.
#
# On approval:  frontend calls execute_tool_call(), then resumes the graph
#               via chatbot.update_state() with the real result.
# On rejection: frontend injects a "cancelled" ToolMessage and resumes.
# =============================================================================
from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage

from ..config import FS_HITL_TOOLS
from ..state import ChatState


def _find_hitl_call(messages: list) -> Optional[dict]:
    """
    Scan the last AIMessage for the first tool call that requires HITL.
    Returns {'tool_name', 'args', 'tool_call_id'} or None.
    """
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            if call["name"] in FS_HITL_TOOLS:
                return {
                    "tool_name":    call["name"],
                    "args":         call["args"],
                    "tool_call_id": call["id"],
                }
        # Last AI message had tool calls but none need HITL → stop.
        return None
    return None


def _build_placeholder(tool_name: str, args: dict) -> str:
    """Build a human-readable approval request message."""
    if tool_name == "write_file":
        path    = args.get("path", "?")
        preview = args.get("content", "")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        return (
            f"⏸️ Pattie wants to write to **{path}**.\n"
            f"Preview of content:\n```\n{preview}\n```\n"
            f"Waiting for your approval…"
        )

    if tool_name == "write_multiple_files":
        files: dict = args.get("files", {})
        file_list   = "\n".join(f"  • {path}" for path in files.keys())
        return (
            f"⏸️ Pattie wants to write **{len(files)} file(s)**:\n"
            f"{file_list}\n"
            f"Waiting for your approval…"
        )

    if tool_name == "delete_file":
        path = args.get("path", "?")
        return (
            f"⏸️ Pattie wants to **delete** `{path}`.\n"
            f"⚠️ This action is irreversible.\n"
            f"Waiting for your approval…"
        )

    if tool_name == "delete_folder":
        path = args.get("path", "?")
        return (
            f"⏸️ Pattie wants to **delete the folder** `{path}` and ALL its contents.\n"
            f"⚠️ This action is irreversible.\n"
            f"Waiting for your approval…"
        )

    return f"⏸️ Pattie wants to run `{tool_name}`. Waiting for your approval…"


def fs_hitl_node(state: ChatState) -> dict:
    """
    Intercept destructive filesystem tool calls and pause for user approval.
    """
    pending = _find_hitl_call(state["messages"])
    if pending is None:
        return {}

    placeholder = _build_placeholder(pending["tool_name"], pending["args"])

    return {
        "messages": [ToolMessage(
            content=placeholder,
            tool_call_id=pending["tool_call_id"],
        )],
        "fs_hitl_pending": pending,   # frontend reads this to show approval widget
    }
