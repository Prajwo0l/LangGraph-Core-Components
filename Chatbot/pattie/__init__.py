# pattie/__init__.py
# Public surface of the Pattie package — import everything the frontend needs from here.

from .graph import build_graph, CHATBOT_CONFIG_DEFAULTS
from .state import ChatState
from .rag import ingest_pdf, set_active_thread, thread_has_document, thread_document_metadata
from .persistence import checkpointer, retreive_all_threads, delete_thread
from .tools.registry import tools, TOOL_GROUPS
from .tools.executor import execute_tool_call
from .memory import (
    get_stm_summary, clear_stm,
    get_all_ltm_facts, get_ltm_profile,
    delete_ltm_fact, clear_all_ltm,
)

__all__ = [
    "build_graph", "CHATBOT_CONFIG_DEFAULTS",
    "ChatState",
    "ingest_pdf", "set_active_thread", "thread_has_document", "thread_document_metadata",
    "checkpointer", "retreive_all_threads", "delete_thread",
    "tools", "TOOL_GROUPS",
    "execute_tool_call",
    "get_stm_summary", "clear_stm",
    "get_all_ltm_facts", "get_ltm_profile", "delete_ltm_fact", "clear_all_ltm",
]
