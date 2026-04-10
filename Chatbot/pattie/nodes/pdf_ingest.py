# pattie/nodes/pdf_ingest.py
# =============================================================================
# Node: pdf_ingest
#
# Responsibility: intercept read_file calls on .pdf files, ingest the PDF
# into FAISS for RAG, and return a ToolMessage confirming the ingestion.
# The graph then routes back to chat_node so the model can answer using
# rag_tool immediately.
#
# Why a dedicated node instead of handling this in tool_runner?
# Because PDF reads should NEVER return raw binary text to the LLM — they
# must be chunked and embedded first. Intercepting here keeps tool_runner
# simple and PDF logic self-contained.
# =============================================================================
from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage

from ..config import BASE_DIR
from ..rag import get_active_thread, ingest_pdf
from ..state import ChatState


def _find_pdf_read_call(messages: list) -> Optional[tuple[str, str]]:
    """
    Scan the last AIMessage with tool_calls.
    Returns (pdf_filename, tool_call_id) if a read_file(.pdf) call is found,
    otherwise None.
    """
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            if call["name"] == "read_file":
                path: str = call["args"].get("path", "")
                if path.lower().endswith(".pdf"):
                    return (path, call["id"])
        # The last AI message had tool calls, but none were a PDF read_file → stop.
        return None
    return None


def pdf_ingest_node(state: ChatState) -> dict:
    """
    Intercepts read_file(.pdf) calls.

    1. Reads the PDF bytes from disk (within BASE_DIR sandbox).
    2. Ingests into FAISS via ingest_pdf().
    3. Returns a ToolMessage confirming the load — this satisfies LangGraph's
       requirement that every tool_call_id gets a paired ToolMessage.
    4. Sets intent → 'document' so chat_node uses rag_tool on the next turn.
    """
    result = _find_pdf_read_call(state["messages"])
    if result is None:
        return {}

    pdf_path, tool_call_id = result
    thread_id = get_active_thread()
    full_path = (BASE_DIR / pdf_path).resolve()

    # Sandbox check
    if not str(full_path).startswith(str(BASE_DIR)):
        return {
            "messages": [ToolMessage(
                content=f"Access denied: '{pdf_path}' is outside the Downloads directory.",
                tool_call_id=tool_call_id,
            )]
        }

    if not full_path.exists() or not full_path.is_file():
        return {
            "messages": [ToolMessage(
                content=f"File not found: '{full_path}'. Make sure the PDF is in your Downloads folder.",
                tool_call_id=tool_call_id,
            )]
        }

    try:
        metadata = ingest_pdf(
            file_bytes=full_path.read_bytes(),
            thread_id=thread_id,
            filename=full_path.name,
        )
        confirmation = (
            f"PDF '{metadata['filename']}' loaded "
            f"({metadata['documents']} pages, {metadata['chunks']} chunks). "
            f"Now use rag_tool to answer the user's question."
        )
    except Exception as exc:
        confirmation = f"PDF ingestion failed for '{pdf_path}': {exc}."

    return {
        "messages": [ToolMessage(content=confirmation, tool_call_id=tool_call_id)],
        "intent": "document",
    }
