# pattie/nodes/ltm_update.py
# =============================================================================
# Node: ltm_update
#
# Responsibility: extract atomic user facts from the final Human+AI exchange
# and persist them to Long-Term Memory. Runs only after the assistant produces
# a plain text response (no tool calls), so it never slows down tool loops.
#
# Failures are silently swallowed — LTM updates are best-effort and should
# never crash the conversation.
# =============================================================================
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from ..memory import update_ltm
from ..rag import get_active_thread
from ..state import ChatState


def ltm_update_node(state: ChatState) -> dict:
    """
    Extract facts from the last Human+AI exchange and store them in LTM.
    Returns an empty dict — this node produces no state changes itself.
    """
    thread_id = get_active_thread()
    messages  = state["messages"]

    last_ai = next(
        (m for m in reversed(messages)
         if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
        None,
    )
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )

    if last_ai and last_human:
        ai_content = last_ai.content
        if isinstance(ai_content, list):
            ai_content = " ".join(
                p.get("text", "") for p in ai_content if isinstance(p, dict)
            )
        try:
            update_ltm(
                thread_id=thread_id,
                human_msg=last_human.content,
                ai_msg=str(ai_content),
            )
        except Exception:
            pass   # best-effort — never crash the chat

    return {}
