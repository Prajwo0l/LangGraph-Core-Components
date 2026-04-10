# pattie/nodes/intent_router.py
# =============================================================================
# Node: intent_router
#
# Responsibility: classify the latest user message into one of the known
# intents so that chat_node binds only the relevant tools.
#
# This node is a PURE ROUTER — it never calls an LLM that generates a reply
# to the user. It only updates state['intent'].
# =============================================================================
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..models import router_llm
from ..rag import get_active_thread, thread_has_document
from ..state import ChatState

# ── Valid intents ─────────────────────────────────────────────────────────────
INTENTS = ["expense", "filesystem", "search", "document", "finance", "general"]

# ── System prompt for the classifier LLM ─────────────────────────────────────
_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a personal AI assistant called Pattie.
Classify the user message into EXACTLY one of these intents:

- expense    : add, list, edit, delete, or summarize expenses, budgets, or income
- filesystem : list, read, write, or delete a file or folder on their computer;
               includes reading multiple files at once or writing multiple files at once
- search     : search the web or find current information online
- document   : ask a question about a PDF or document already loaded in this session
- finance    : stock prices or arithmetic calculations
- general    : casual conversation, greetings, or anything that does not fit above

CRITICAL RULES:

1. "summarize the pdf / explain the document / what does the pdf say" → ALWAYS "document".
   The user already uploaded it; no filesystem access is needed.

2. "open report.pdf from my Downloads / read these three files / write these files" → "filesystem".

3. "document" intent = asking about content of an already-loaded/uploaded file.
   "filesystem" intent = explicitly accessing, listing, reading, or writing files on disk,
   including batch operations like 'read these files' or 'create these files'.

4. If unsure between "filesystem" and "document", choose "document" if a file was
   already mentioned as uploaded in this session.

Reply with ONLY the intent word. No explanation. No punctuation.\
"""

# ── Keyword heuristics (fast path, no LLM needed) ────────────────────────────
_DOC_TRIGGERS = (
    "summarize", "summary", "explain", "describe", "what does",
    "what is", "tell me about", "extract", "who", "when", "where",
    "how many", "pdf", "document", "resume", "cv", "uploaded", "the file", "this file",
)
_FS_TRIGGERS = (
    "download", "downloads folder", "open the file", "read the file",
    "read these files", "read multiple", "from my computer", "from downloads",
    "write to", "write these files", "write multiple", "create these files", "delete",
)


def intent_router(state: ChatState) -> dict:
    """
    Classify the latest human message and store the result in state['intent'].
    Uses keyword fast-paths when possible; falls back to an LLM classifier.
    """
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    msg_lower = last_human.lower()
    doc_loaded = thread_has_document(get_active_thread())
    wants_fs   = any(t in msg_lower for t in _FS_TRIGGERS)
    wants_doc  = any(t in msg_lower for t in _DOC_TRIGGERS)

    # Fast path: doc already loaded and message clearly refers to it
    if doc_loaded and wants_doc and not wants_fs:
        return {"intent": "document"}

    # LLM classifier
    response = router_llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_human),
    ])
    intent = response.content.strip().lower()
    if intent not in INTENTS:
        intent = "general"

    # Safety override: doc loaded + LLM picked wrong intent for a doc-like query
    if doc_loaded and intent in ("filesystem", "general") and wants_doc and not wants_fs:
        intent = "document"

    return {"intent": intent}
