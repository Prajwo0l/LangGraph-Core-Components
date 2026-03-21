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
from typing import Annotated, Any, Dict, Optional

# ── Third-party ──
import requests
from dotenv import load_dotenv
from ddgs import DDGS

# ── LangChain / LangGraph ──
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from typing import TypedDict

# ── Environment ──
load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'Personal Chatbot'
warnings.filterwarnings('ignore', message='could not convert string to float')


# =============================================================================
# State
# =============================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str
    intent: str  # set by intent_router: expense | search | document | finance | general

# Default state values for new threads
DEFAULT_STATE: dict = {'title': 'New Chat', 'intent': 'general'}


# =============================================================================
# Models
# =============================================================================
llm = ChatOpenAI(model='gpt-4o-mini', max_retries=2)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')


# =============================================================================
# RAG  —  per-thread PDF retriever store
# =============================================================================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
_ACTIVE_THREAD_ID: Optional[str] = None


def set_active_thread(thread_id: str) -> None:
    """Call this before every chat stream so rag_tool uses the right retriever."""
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
_db_conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=_db_conn)


# =============================================================================
# Tools
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
            f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
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
    # Whitelist: only allow digits, operators, spaces, dots, parentheses
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
    Retrieve relevant passages from the PDF document uploaded in this chat.
    Use this whenever the user asks questions about an uploaded file or document.
    """
    retriever = _THREAD_RETRIEVERS.get(_ACTIVE_THREAD_ID)
    if retriever is None:
        return {'error': 'No document uploaded yet. Please upload a PDF first.'}
    docs = retriever.invoke(query)
    return {
        'query': query,
        'context': [d.page_content for d in docs],
        'source_file': _THREAD_METADATA.get(_ACTIVE_THREAD_ID, {}).get('filename'),
    }


# =============================================================================
# MCP tools  (Expense Tracker)
# =============================================================================
async def _load_mcp_tools() -> list:
    client = MultiServerMCPClient({
        'expense_tracker': {
            'command': r'C:\Users\lamic\Desktop\Expense MCP Server\.venv\Scripts\python.exe',
            'args': [r'C:\Users\lamic\Desktop\Expense MCP Server\main.py'],
            'transport': 'stdio',
            'cwd': r'C:\Users\lamic\Desktop\Expense MCP Server',
        }
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

# All tools flat list (used by tool_node executor)
tools = [search_tool, calculator, get_stock_price, rag_tool] + mcp_tools

# Pull out mcp tools by name for grouping
_mcp_by_name = {t.name: t for t in mcp_tools}

# Tool groups — each intent gets only the tools it needs
TOOL_GROUPS: Dict[str, list] = {
    # expense: add/list/summarize expenses + calendar
    'expense': [t for t in mcp_tools],

    # search: web search only
    'search': [search_tool],

    # document: RAG over uploaded PDF
    'document': [rag_tool],

    # finance: stock prices + calculator
    'finance': [get_stock_price, calculator],

    # general: no tools — pure conversation
    'general': [],
}

# Pre-bind each group so we don't re-bind on every call
_llm_by_intent: Dict[str, Any] = {
    intent: llm.bind_tools(tool_list) if tool_list else llm
    for intent, tool_list in TOOL_GROUPS.items()
}

# Fallback — all tools bound (used if intent is unknown)
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# Graph nodes
# =============================================================================

# Fast small LLM just for intent classification
_router_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_retries=2)

INTENTS = ['expense', 'search', 'document', 'finance', 'general']

def intent_router(state: ChatState) -> dict:
    """
    Classifies the latest user message into one of five intents.
    This is a lightweight LLM call — no tools, just classification.

    expense  — adding/listing/summarizing expenses
    search   — searching the web for information
    document — asking questions about an uploaded PDF
    finance  — stock prices or math calculations
    general  — casual chat, greetings, anything else
    """
    last_human = next(
        (m.content for m in reversed(state['messages']) if isinstance(m, HumanMessage)),
        ''
    )

    classification = _router_llm.invoke([
        SystemMessage(content=(
            'You are an intent classifier. Classify the user message into EXACTLY one of these intents:\n'
            '- expense: user wants to add, list, or summarize expenses\n'
            '- search: user wants to search the web or find current information\n'
            '- document: user is asking about an uploaded PDF or document\n'
            '- finance: user wants stock prices or math calculations\n'
            '- general: casual conversation, greetings, or anything else\n\n'
            'Reply with ONLY the intent word. No explanation. No punctuation.'
        )),
        HumanMessage(content=last_human),
    ])

    intent = classification.content.strip().lower()
    if intent not in INTENTS:
        intent = 'general'

    return {'intent': intent}


def chat_node(state: ChatState) -> dict:
    """Main LLM node — uses only the tools relevant to the detected intent."""
    today = date.today().strftime('%Y-%m-%d')
    intent = state.get('intent', 'general')

    # Pick the right pre-bound LLM for this intent
    active_llm = _llm_by_intent.get(intent, llm_with_tools)

    # System prompt tailored per intent
    intent_hints = {
        'expense':  'You are helping the user track expenses. Use expense tools to add, list or summarize.',
        'search':   'You are helping the user find information. Use the search tool to look things up.',
        'document': 'You are helping the user understand an uploaded PDF. Use the rag_tool to retrieve context.',
        'finance':  'You are helping the user with financial data. Use get_stock_price or calculator as needed.',
        'general':  'You are having a friendly conversation. No tools needed unless the user asks explicitly.',
    }

    system = SystemMessage(content=(
        f'You are Pattie, a helpful personal AI assistant.\n'
        f"Today's date is {today}.\n"
        f'When adding expenses, use {today} if the user does not specify a date.\n'
        f'Always pass all required arguments when calling tools.\n'
        f'Current task: {intent_hints.get(intent, "")}'
    ))

    response = active_llm.invoke([system] + state['messages'])
    return {'messages': [response]}


async def _invoke_tool(t: Any, args: dict) -> str:  # exported for direct use by frontend
    """Invoke a tool async-first, fall back to sync. Unwrap MCP content blocks."""
    try:
        raw = await t.ainvoke(args)
    except NotImplementedError:
        raw = t.invoke(args)

    if isinstance(raw, list):          # MCP returns [{type, text, id}, ...]
        return ' '.join(
            b.get('text', str(b)) for b in raw if isinstance(b, dict)
        )
    if isinstance(raw, dict):
        return json.dumps(raw)
    return str(raw)


def tool_node(state: ChatState) -> dict:
    """Execute every tool call in the last AI message."""
    last = state['messages'][-1]
    results = []
    for call in last.tool_calls:
        matched = next((t for t in tools if t.name == call['name']), None)
        if matched is None:
            content = f"Tool '{call['name']}' not found."
        else:
            try:
                content = asyncio.run(_invoke_tool(matched, call['args']))
            except Exception as exc:
                content = f'Tool error: {exc}'
        results.append(ToolMessage(content=content, tool_call_id=call['id']))
    return {'messages': results}


# =============================================================================
# Graph
# =============================================================================
_graph = StateGraph(ChatState)

# Nodes
_graph.add_node('intent_router', intent_router)
_graph.add_node('chat_node', chat_node)
_graph.add_node('tools', tool_node)

# Edges
# 1. Every message starts with intent classification
_graph.add_edge(START, 'intent_router')

# 2. After routing, always go to chat_node
_graph.add_edge('intent_router', 'chat_node')

# 3. After chat_node: go to tools if there are tool calls, else END
_graph.add_conditional_edges('chat_node', tools_condition)

# 4. After tools: back to chat_node (NOT intent_router — intent is already set)
_graph.add_edge('tools', 'chat_node')

chatbot = _graph.compile(checkpointer=checkpointer)
CHATBOT_CONFIG_DEFAULTS = {'recursion_limit': 15}


# =============================================================================
# Thread helpers
# =============================================================================
def retreive_all_threads() -> dict:
    """Return {thread_id: title} ordered oldest → newest."""
    seen: Dict[str, str] = {}
    for cp in checkpointer.list(None):
        tid = cp.config['configurable'].get('thread_id')
        if tid not in seen:
            seen[tid] = cp.checkpoint.get('channel_values', {}).get('title', 'New Chat')
    return dict(reversed(list(seen.items())))


def delete_thread(thread_id: str) -> None:
    """Delete all checkpoints for a thread from SQLite."""
    tables = _db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    for (table,) in tables:
        cols = [r[1] for r in _db_conn.execute(f'PRAGMA table_info({table})').fetchall()]
        if 'thread_id' in cols:
            _db_conn.execute(f'DELETE FROM {table} WHERE thread_id = ?', (thread_id,))
    _db_conn.commit()