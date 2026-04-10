# pattie/nodes/tool_runner.py
# =============================================================================
# Node: tool_runner
#
# Responsibility: execute every non-HITL tool call that appears in the last
# AIMessage and return ToolMessages paired to their tool_call_ids.
#
# CRITICAL: every tool_call_id in an AIMessage MUST receive a paired
# ToolMessage before the graph ends — OpenAI returns HTTP 400 otherwise.
# =============================================================================
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, ToolMessage

from ..state import ChatState
from ..tools.executor import invoke_tool_async
from ..tools.registry import tools_by_name


def tool_runner(state: ChatState) -> dict:
    """Execute all tool calls in the last AIMessage and return ToolMessages."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {}

    results = []
    for call in last.tool_calls:
        tool_obj = tools_by_name.get(call["name"])
        if tool_obj is None:
            content = f"Tool '{call['name']}' not found."
        else:
            try:
                content = asyncio.run(invoke_tool_async(tool_obj, call["args"]))
            except Exception as exc:
                content = f"Tool error: {exc}"
        results.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return {"messages": results}
