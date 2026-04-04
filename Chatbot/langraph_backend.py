# =============================================================================
# langraph_backend.py  —  Pattie AI Assistant Backend
# =============================================================================
from __future__ import annotations

# ── Standard library ──
import asyncio
import json
import os
import sqlite3
import tempfile
import warnings
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

# ── Third-party ──
import requests
from dotenv import load_dotenv
from ddgs import DDGS

# ── LangChain / LangGraph ──
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict

# ── Memory (STM + LTM) ──
from memory import (
    apply_stm,
    build_memory_context,
    update_ltm,
    get_stm_summary,
    clear_stm,
    get_all_ltm_facts,
    get_ltm_profile,
    delete_ltm_fact,
    clear_all_ltm,
)

# Re-export memory helpers so streamlit_frontend only needs to import from langraph_backend
__all_memory__ = [
    'get_stm_summary', 'clear_stm',
    'get_all_ltm_facts', 'get_ltm_profile', 'delete_ltm_fact', 'clear_all_ltm',
]

# ── Environment ──
load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'Personal Chatbot'
warnings.filterwarnings('ignore', message='could not convert string to float')

# ── Base directory — must match the MCP filesystem server's BASE_DIR ──
BASE_DIR = Path(r'C:\Users\lamic\Downloads').resolve()

# ── Filesystem tools that require human approval before executing ──
FS_HITL_TOOLS = {'write_file', 'delete_file', 'delete_folder'}

# ── Filesystem tools that are safe to run without approval ──
FS_AUTO_TOOLS = {'list_files', 'read_file'}


# =============================================================================
# State
# =============================================================================
class ChatState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    title:          str
    intent:         str   # expense | filesystem | search | document | finance | general
    # Holds a pending filesystem tool call that needs user approval.
    # Structure: {'tool_name': str, 'args': dict, 'tool_call_id': str}
    # None means nothing is pending.
    fs_hitl_pending: Optional[dict]

DEFAULT_STATE: dict = {'title': 'New Chat', 'intent': 'general', 'fs_hitl_pending': None}


# =============================================================================
# Models
# =============================================================================
llm = ChatOpenAI(model='gpt-4o-mini', max_retries=2)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')


# =============================================================================
# RAG  —  per-thread PDF retriever store
# =============================================================================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA:   Dict[str, dict] = {}
_ACTIVE_THREAD_ID:  Optional[str]   = None


def set_active_thread(thread_id: str) -> None:
    global _ACTIVE_THREAD_ID
    _ACTIVE_THREAD_ID = str(thread_id)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Chunk a PDF, embed it, and store the retriever keyed by thread_id."""
    if not file_bytes:
        raise ValueError('No bytes received for ingestion.')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=['\n\n', '\n', ' ', '']
        ).split_documents(docs)
        retriever = FAISS.from_documents(chunks, embeddings).as_retriever(
            search_type='similarity', search_kwargs={'k': 4}
        )
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            'filename': filename or os.path.basename(tmp_path),
            'documents': len(docs),
            'chunks': len(chunks),
        }
        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# =============================================================================
# SQLite checkpointer
# =============================================================================
_db_conn   = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=_db_conn)


# =============================================================================
# LangChain tools (non-MCP)
# =============================================================================
@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo. Use for current events or factual lookups."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='us-en', max_results=5))
        if not results:
            return 'No results found.'
        return '\n\n'.join(
            f"Title: {r.get('title','')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')}"
            for r in results
        )
    except Exception as exc:
        return f'Search error: {exc}'


@tool
def calculator(expression: str) -> dict:
    """
    Safely evaluate a basic arithmetic expression (+, -, *, /, **, %).
    Example: '396.73 * 50'  or  '(100 + 200) / 3'
    """
    allowed = set('0123456789+-*/.() %**\t\n')
    if not all(c in allowed for c in expression):
        return {'error': 'Expression contains invalid characters.'}
    try:
        result = float(eval(expression, {'__builtins__': {}}, {}))
        return {'expression': expression, 'result': result}
    except Exception as exc:
        return {'error': str(exc)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a ticker symbol (e.g. AAPL, TSLA, AMZN)."""
    url = (
        f'https://www.alphavantage.co/query'
        f'?function=GLOBAL_QUOTE&symbol={symbol}&apikey=CE8MY894ND1ESUK5'
    )
    try:
        data = requests.get(url, timeout=10).json()
        price = float(data['Global Quote']['05. price'])
        return {'symbol': symbol, 'price': price}
    except Exception:
        return {'error': f'Could not fetch stock price for {symbol}.'}


@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant passages from the PDF document that is currently loaded
    in this session (either uploaded by the user or opened from the filesystem).
    Use this whenever the user asks questions about a document or PDF.
    """
    retriever = _THREAD_RETRIEVERS.get(_ACTIVE_THREAD_ID)
    if retriever is None:
        return {'error': 'No document loaded yet. Ask the user to upload a PDF or open one from their Downloads folder.'}
    docs = retriever.invoke(query)
    return {
        'query': query,
        'context': [d.page_content for d in docs],
        'source_file': _THREAD_METADATA.get(_ACTIVE_THREAD_ID, {}).get('filename'),
    }


# =============================================================================
# MCP tools  (Expense Tracker + Filesystem)
# =============================================================================
async def _load_mcp_tools() -> list:
    client = MultiServerMCPClient({
        'expense_tracker': {
            'command': r'C:\Users\lamic\Desktop\Expense MCP Server\.venv\Scripts\python.exe',
            'args':    [r'C:\Users\lamic\Desktop\Expense MCP Server\main.py'],
            'transport': 'stdio',
            'cwd':     r'C:\Users\lamic\Desktop\Expense MCP Server',
        },
        'filesystem': {
            'command': r'C:\Users\lamic\Desktop\File-System-MCP-Server\.venv\Scripts\python.exe',
            'args':    [r'C:\Users\lamic\Desktop\File-System-MCP-Server\main.py'],
            'transport': 'stdio',
            'cwd':     r'C:\Users\lamic\Desktop\File-System-MCP-Server',
        },
    })
    return await client.get_tools()


try:
    mcp_tools = asyncio.run(_load_mcp_tools())
except Exception as exc:
    print(f'[WARNING] Could not load MCP tools: {exc}')
    mcp_tools = []


# =============================================================================
# Tool registry — grouped by intent
# =============================================================================
tools = [search_tool, calculator, get_stock_price, rag_tool] + mcp_tools

# Named lookup used by tool execution helpers
_tools_by_name: Dict[str, Any] = {t.name: t for t in tools}

_expense_tools = [t for t in mcp_tools if t.name in {
    'add_expense', 'add_to_calendar', 'list_expenses', 'summarize',
    'set_budget', 'list_budgets', 'check_budget_alerts', 'delete_budget',
    'add_credit', 'list_credits', 'edit_credit', 'delete_credit',
    'edit_expense', 'delete_expense', 'monthly_overview',
}]
_filesystem_tools = [t for t in mcp_tools if t.name in {
    'list_files', 'read_file', 'write_file', 'delete_file', 'delete_folder',
}]

TOOL_GROUPS: Dict[str, list] = {
    'expense':    _expense_tools,
    'filesystem': _filesystem_tools,
    'search':     [search_tool],
    'document':   [rag_tool],
    'finance':    [get_stock_price, calculator],
    'general':    [],
}

_llm_by_intent: Dict[str, Any] = {
    intent: llm.bind_tools(tool_list) if tool_list else llm
    for intent, tool_list in TOOL_GROUPS.items()
}

llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# Intent router
# =============================================================================
_router_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_retries=2)

INTENTS = ['expense', 'filesystem', 'search', 'document', 'finance', 'general']

_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a personal AI assistant called Pattie.
Classify the user message into EXACTLY one of these intents:

- expense    : user wants to add, list, edit, delete, or summarize expenses, budgets, or income
- filesystem : user wants to list, read/open, write, delete, or access a file or folder on their computer
- search     : user wants to search the web or find current information online
- document   : user is asking a question about a PDF or document (uploaded or already loaded in this session)
- finance    : user wants stock prices or math / arithmetic calculations
- general    : casual conversation, greetings, or anything that does not fit the above

CRITICAL RULES:

1. If the user says things like "summarize the pdf", "explain the document",
   "what does the pdf say", "what is in the uploaded file", "tell me about my resume",
   "summarize what I uploaded" — intent is ALWAYS "document".
   The user already uploaded it; no filesystem access is needed.

2. If the user mentions a specific filename AND asks to fetch it from their computer
   (e.g. "open report.pdf from my Downloads") → intent is "filesystem".

3. "document" intent is correct when:
   - The user refers to content of an already-uploaded/opened file.
   - The user uses words like: summarize, explain, describe, extract, what does it say, who is.

4. "filesystem" intent is correct when:
   - The user explicitly wants to access/list/write/delete files on their computer.
   - The user mentions a path, folder, or "Downloads".

5. If unsure between "filesystem" and "document", choose "document" if a file
   was already mentioned as uploaded in this session.

Reply with ONLY the intent word. No explanation. No punctuation.\
"""


def intent_router(state: ChatState) -> dict:
    last_human = next(
        (m.content for m in reversed(state['messages']) if isinstance(m, HumanMessage)),
        ''
    )

    # ── Short-circuit: if a PDF is already loaded for this thread,
    #    and the user isn't asking about a NEW file → force 'document'
    active_tid = _ACTIVE_THREAD_ID or 'default'
    doc_loaded = thread_has_document(active_tid)

    # Keywords that suggest the user wants to work with the already-loaded doc
    _DOC_TRIGGERS = (
        'summarize', 'summary', 'explain', 'describe', 'what does',
        'what is', 'tell me about', 'extract', 'list', 'who', 'when',
        'where', 'how many', 'pdf', 'document', 'resume', 'cv',
        'uploaded', 'the file', 'this file',
    )
    # Keywords that suggest a NEW filesystem operation
    _FS_TRIGGERS = (
        'download', 'downloads folder', 'open the file', 'read the file',
        'from my computer', 'from downloads', 'write to', 'delete',
    )

    msg_lower = last_human.lower()
    wants_fs  = any(t in msg_lower for t in _FS_TRIGGERS)
    wants_doc = any(t in msg_lower for t in _DOC_TRIGGERS)

    if doc_loaded and wants_doc and not wants_fs:
        return {'intent': 'document'}

    classification = _router_llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_human),
    ])
    intent = classification.content.strip().lower()
    if intent not in INTENTS:
        intent = 'general'

    # ── Safety override: if doc is loaded and LLM picks filesystem/general
    #    for a doc-like query (no explicit FS trigger), redirect to document
    if doc_loaded and intent in ('filesystem', 'general') and wants_doc and not wants_fs:
        intent = 'document'

    return {'intent': intent}


# =============================================================================
# Tool invocation helper
# =============================================================================
async def _invoke_tool(t: Any, args: dict) -> str:
    """Invoke a tool (async-first, sync fallback). Unwrap MCP content blocks."""
    try:
        raw = await t.ainvoke(args)
    except NotImplementedError:
        raw = t.invoke(args)

    if isinstance(raw, list):
        return ' '.join(b.get('text', str(b)) for b in raw if isinstance(b, dict))
    if isinstance(raw, dict):
        return json.dumps(raw)
    return str(raw)


def execute_tool_call(tool_name: str, args: dict) -> str:
    """
    Public helper — execute a HITL-approved tool synchronously.
    Used by the frontend after the user approves a write_file or delete_file action.

    WHY we bypass _invoke_tool for MCP filesystem tools:
      MCP tools communicate over a stdio subprocess that was opened once at startup
      inside asyncio.run(_load_mcp_tools()). That event loop is gone by the time
      the frontend calls this function. Calling asyncio.run() again creates a NEW
      loop with no live MCP connection, so the MCP call silently returns nothing —
      write_file appears to work (file was already written by the graph), but
      delete_file does nothing because the MCP subprocess isn't reachable.

      Solution: for the two filesystem HITL tools we perform the operation DIRECTLY
      in Python using pathlib — same sandbox logic as the MCP server, no subprocess needed.
    """
    # ── Direct pathlib execution for filesystem HITL tools ───────────────────
    if tool_name == 'delete_file':
        path_arg = args.get('path', '')
        try:
            p = (BASE_DIR / path_arg).resolve()
            if not str(p).startswith(str(BASE_DIR)):
                return f"Access denied: '{path_arg}' is outside the Downloads directory."
            if not p.exists():
                return f"File does not exist: {path_arg}"
            if not p.is_file():
                return f"'{path_arg}' is a directory — use delete_folder to remove directories."
            p.unlink()
            return f"File deleted: {path_arg}"
        except Exception as exc:
            return f"delete_file error: {exc}"

    if tool_name == 'delete_folder':
        import shutil
        path_arg = args.get('path', '')
        try:
            p = (BASE_DIR / path_arg).resolve()
            if not str(p).startswith(str(BASE_DIR)):
                return f"Access denied: '{path_arg}' is outside the Downloads directory."
            if not p.exists():
                return f"Folder does not exist: {path_arg}"
            if not p.is_dir():
                return f"'{path_arg}' is a file — use delete_file to remove files."
            shutil.rmtree(p)
            return f"Folder deleted: {path_arg}"
        except Exception as exc:
            return f"delete_folder error: {exc}"

    if tool_name == 'write_file':
        path_arg = args.get('path', '')
        content  = args.get('content', '')
        try:
            p = (BASE_DIR / path_arg).resolve()
            if not str(p).startswith(str(BASE_DIR)):
                return f"Access denied: '{path_arg}' is outside the Downloads directory."
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return f"File written successfully: {p}"
        except Exception as exc:
            return f"write_file error: {exc}"

    # ── Fallback for any other tool (non-MCP LangChain tools) ────────────────
    t = _tools_by_name.get(tool_name)
    if t is None:
        return json.dumps({'error': f"Tool '{tool_name}' not found."})
    try:
        return asyncio.run(_invoke_tool(t, args))
    except Exception as exc:
        return json.dumps({'error': str(exc)})


# =============================================================================
# PDF interception node  (read_file on .pdf → FAISS ingestion)
# =============================================================================

def _find_pdf_read_call(messages: list[BaseMessage]) -> Optional[tuple[str, str]]:
    """
    Scan the last AIMessage with tool_calls.
    Returns (pdf_filename, tool_call_id) if a read_file(.pdf) call is found.
    """
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        tool_calls = getattr(m, 'tool_calls', None)
        if not tool_calls:
            continue
        for call in tool_calls:
            if call['name'] == 'read_file':
                path: str = call['args'].get('path', '')
                if path.lower().endswith('.pdf'):
                    return (path, call['id'])
        return None   # last AI msg had tool calls but no PDF read_file → stop
    return None


def pdf_ingest_node(state: ChatState) -> dict:
    """
    Intercepts read_file(.pdf) calls.
    Reads bytes from disk → ingests into FAISS → returns a ToolMessage
    with a confirmation so LangGraph pairs it with the tool_call_id.
    Sets intent → 'document' for the next chat_node turn.
    """
    result = _find_pdf_read_call(state['messages'])
    if result is None:
        return {}

    pdf_path, tool_call_id = result
    thread_id  = _ACTIVE_THREAD_ID or 'default'
    full_path  = (BASE_DIR / pdf_path).resolve()

    if not str(full_path).startswith(str(BASE_DIR)):
        return {
            'messages': [ToolMessage(
                content=f"Access denied: '{pdf_path}' is outside the Downloads directory.",
                tool_call_id=tool_call_id,
            )],
        }

    if not full_path.exists() or not full_path.is_file():
        return {
            'messages': [ToolMessage(
                content=f"File not found: '{full_path}'. Make sure the PDF is in your Downloads folder.",
                tool_call_id=tool_call_id,
            )],
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
        'messages': [ToolMessage(content=confirmation, tool_call_id=tool_call_id)],
        'intent':   'document',
    }


# =============================================================================
# Filesystem HITL node  (write_file / delete_file → pause for approval)
# =============================================================================

def _find_fs_hitl_call(messages: list[BaseMessage]) -> Optional[dict]:
    """
    Scan the last AIMessage with tool_calls.
    Returns the FIRST call that requires HITL approval, or None.
    Returns: {'tool_name': str, 'args': dict, 'tool_call_id': str}
    """
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        tool_calls = getattr(m, 'tool_calls', None)
        if not tool_calls:
            continue
        for call in tool_calls:
            if call['name'] in FS_HITL_TOOLS:
                return {
                    'tool_name':    call['name'],
                    'args':         call['args'],
                    'tool_call_id': call['id'],
                }
        return None   # last AI msg had calls but none need HITL
    return None


def fs_hitl_node(state: ChatState) -> dict:
    """
    Intercepts write_file / delete_file calls.

    Instead of executing them immediately, this node:
      1. Stores the pending call in state['fs_hitl_pending'].
      2. Returns a placeholder ToolMessage saying "Awaiting your approval".
         This satisfies LangGraph's requirement that every tool_call_id
         gets a paired ToolMessage before the graph can end the turn.
      3. The graph then ends — the frontend reads fs_hitl_pending from
         the LangGraph state and renders the approval widget.

    On approval:  frontend calls execute_tool_call() directly, then
                  injects an updated ToolMessage into the graph via
                  chatbot.update_state() and resumes streaming.
    On rejection: frontend injects a "cancelled" ToolMessage and resumes.
    """
    pending = _find_fs_hitl_call(state['messages'])
    if pending is None:
        return {}

    tool_name    = pending['tool_name']
    args         = pending['args']
    tool_call_id = pending['tool_call_id']

    # Build a human-readable description for the placeholder
    if tool_name == 'write_file':
        desc = f"write to **{args.get('path', '?')}**"
        preview = args.get('content', '')
        if len(preview) > 120:
            preview = preview[:120] + '…'
        placeholder_msg = (
            f"⏸️ Pattie wants to {desc}.\n"
            f"Preview of content:\n```\n{preview}\n```\n"
            f"Waiting for your approval…"
        )
    else:  # delete_file
        desc = f"delete **{args.get('path', '?')}**"
        placeholder_msg = (
            f"⏸️ Pattie wants to {desc}.\n"
            f"⚠️ This action is irreversible.\n"
            f"Waiting for your approval…"
        )

    return {
        'messages': [ToolMessage(
            content=placeholder_msg,
            tool_call_id=tool_call_id,
        )],
        'fs_hitl_pending': pending,   # stored in graph state for frontend to read
    }


# =============================================================================
# Chat node
# =============================================================================

def chat_node(state: ChatState) -> dict:
    """Main LLM node — uses only the tools relevant to the detected intent."""
    today      = date.today().strftime('%Y-%m-%d')
    intent     = state.get('intent', 'general')
    thread_id  = _ACTIVE_THREAD_ID or 'default'

    active_llm = _llm_by_intent.get(intent, llm_with_tools)

    intent_hints = {
        'expense':    'You are helping the user track expenses, budgets, and income. Use expense tools to add, list, edit, or summarize.',
        'filesystem': (
            'You are helping the user manage files in their Downloads folder. '
            'Available tools: list_files (browse), read_file (open text files), '
            'write_file (create/overwrite), delete_file (remove). '
            'For .pdf files, call read_file with just the filename — PDF ingestion is automatic. '
            'Use only the filename relative to Downloads, not the full path.'
        ),
        'search':     'You are helping the user find information. Use the search tool to look things up.',
        'document':   (
            'A PDF document has been loaded into memory. '
            'Use rag_tool with the user\'s question to retrieve relevant passages and answer them.'
        ),
        'finance':    'You are helping with financial data. Use get_stock_price or calculator as needed.',
        'general':    'You are having a friendly conversation. No tools needed unless the user explicitly asks.',
    }

    # ── Apply STM: compress old messages, keep last 8 recent ──────────────────
    stm_messages = apply_stm(thread_id, state['messages'])

    # ── Get last human message for LTM retrieval ──────────────────────────────
    last_human_content = next(
        (m.content for m in reversed(stm_messages) if isinstance(m, HumanMessage)),
        ''
    )

    # ── Build LTM memory context ───────────────────────────────────────────────
    memory_context = build_memory_context(thread_id, stm_messages, last_human_content)

    # ── Compose system message ────────────────────────────────────────────────
    memory_section = f'\n\n{memory_context}' if memory_context else ''

    system = SystemMessage(content=(
        f'You are Pattie, a helpful personal AI assistant.\n'
        f"Today's date is {today}.\n"
        f'When adding expenses, use {today} if the user does not specify a date.\n'
        f'Always pass all required arguments when calling tools.\n'
        f'Current task: {intent_hints.get(intent, "")}'
        f'{memory_section}'
    ))

    response = active_llm.invoke([system] + stm_messages)
    return {'messages': [response]}


# =============================================================================
# Tool node  (executes non-HITL tool calls)
# =============================================================================

def tool_node(state: ChatState) -> dict:
    """Execute every tool call in the last AI message."""
    last = state['messages'][-1]
    if not isinstance(last, AIMessage) or not getattr(last, 'tool_calls', None):
        return {}
    results = []
    for call in last.tool_calls:
        t = _tools_by_name.get(call['name'])
        if t is None:
            content = f"Tool '{call['name']}' not found."
        else:
            try:
                content = asyncio.run(_invoke_tool(t, call['args']))
            except Exception as exc:
                content = f'Tool error: {exc}'
        results.append(ToolMessage(content=content, tool_call_id=call['id']))
    return {'messages': results}


def ltm_update_node(state: ChatState) -> dict:
    """
    After the assistant produces a final response (no more tool calls),
    extract atomic facts from the last exchange and update LTM.
    This runs asynchronously in a background thread so the user sees no delay.
    """
    thread_id = _ACTIVE_THREAD_ID or 'default'
    messages  = state['messages']

    # Find the last Human + last AI message pair
    last_ai = next(
        (m for m in reversed(messages)
         if isinstance(m, AIMessage) and not getattr(m, 'tool_calls', None)),
        None
    )
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None
    )

    if last_ai and last_human:
        ai_content = last_ai.content
        if isinstance(ai_content, list):
            ai_content = ' '.join(
                p.get('text', '') for p in ai_content if isinstance(p, dict)
            )
        try:
            update_ltm(
                thread_id=thread_id,
                human_msg=last_human.content,
                ai_msg=str(ai_content),
            )
        except Exception:
            pass   # LTM update is best-effort; never crash the chat

    return {}   # no state changes


# =============================================================================
# Graph routing
# =============================================================================

def _after_chat(state: ChatState) -> str:
    """
    Decide what happens after chat_node:

      1. No tool calls at all          → ltm_update  (extract facts, then END)
      2. read_file on a .pdf           → pdf_ingest  (bypass tool_node)
      3. write_file / delete_file      → fs_hitl     (pause for approval)
      4. Any other tool call           → tools       (execute normally)

    CRITICAL: every tool_call_id in the last AIMessage MUST get a paired
    ToolMessage before the graph ends, otherwise OpenAI raises a 400.
    All branches here guarantee that.
    """
    last = state['messages'][-1]

    # No tool calls → update LTM then end
    if not isinstance(last, AIMessage) or not getattr(last, 'tool_calls', None):
        return 'ltm_update'

    # PDF read intercept (must check before HITL — read_file on .pdf is auto)
    if _find_pdf_read_call(state['messages']) is not None:
        return 'pdf_ingest'

    # Filesystem HITL (write / delete need approval)
    if _find_fs_hitl_call(state['messages']) is not None:
        return 'fs_hitl'

    # All other tool calls (including any stray read_file on non-PDFs) → tools
    return 'tools'


# =============================================================================
# Graph assembly
# =============================================================================
_graph = StateGraph(ChatState)

_graph.add_node('intent_router', intent_router)
_graph.add_node('chat_node',     chat_node)
_graph.add_node('tools',         tool_node)
_graph.add_node('pdf_ingest',    pdf_ingest_node)
_graph.add_node('fs_hitl',       fs_hitl_node)
_graph.add_node('ltm_update',    ltm_update_node)

# Flow
_graph.add_edge(START, 'intent_router')
_graph.add_edge('intent_router', 'chat_node')

_graph.add_conditional_edges(
    'chat_node',
    _after_chat,
    {END: END, 'ltm_update': 'ltm_update', 'pdf_ingest': 'pdf_ingest', 'fs_hitl': 'fs_hitl', 'tools': 'tools'},
)

_graph.add_edge('ltm_update', END)    # LTM updated → done
_graph.add_edge('tools',      'chat_node')   # normal loop
_graph.add_edge('pdf_ingest', 'chat_node')   # after PDF load → answer from RAG
_graph.add_edge('fs_hitl',    END)           # pause here; frontend resumes after approval

chatbot = _graph.compile(checkpointer=checkpointer)
CHATBOT_CONFIG_DEFAULTS = {'recursion_limit': 25}


# =============================================================================
# Thread helpers
# =============================================================================
def retreive_all_threads() -> dict:
    """
    Retrieve all threads directly from SQLite, bypassing the SqliteSaver.list()
    serializer which is broken in some langgraph-checkpoint-sqlite versions
    (AttributeError: 'JsonPlusSerializer' object has no attribute 'loads').
    """
    seen: Dict[str, str] = {}
    try:
        # Try the normal LangGraph API first
        for cp in checkpointer.list(None):
            tid = cp.config['configurable'].get('thread_id')
            if tid not in seen:
                seen[tid] = cp.checkpoint.get('channel_values', {}).get('title', 'New Chat')
    except (AttributeError, Exception):
        # Fallback: query SQLite directly to list threads + titles
        try:
            rows = _db_conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC"
            ).fetchall()
            for (tid,) in rows:
                if tid in seen:
                    continue
                # Try to extract the title from the checkpoint blob directly
                title = 'New Chat'
                try:
                    blob_row = _db_conn.execute(
                        "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                        (tid,)
                    ).fetchone()
                    if blob_row:
                        import json as _json
                        data = _json.loads(blob_row[0])
                        title = (
                            data.get('channel_values', {}).get('title')
                            or 'New Chat'
                        )
                except Exception:
                    pass
                seen[tid] = title
        except Exception:
            pass
    return dict(reversed(list(seen.items())))


def delete_thread(thread_id: str) -> None:
    tables = _db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    for (table,) in tables:
        cols = [r[1] for r in _db_conn.execute(f'PRAGMA table_info({table})').fetchall()]
        if 'thread_id' in cols:
            _db_conn.execute(f'DELETE FROM {table} WHERE thread_id = ?', (thread_id,))
    _db_conn.commit()
