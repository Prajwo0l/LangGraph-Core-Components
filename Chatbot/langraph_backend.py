from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph ,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from ddgs import DDGS
import asyncio
import requests
import random



os.environ['LANGCHAIN_PROJECT']='Personal Chatbot'
load_dotenv()

# ------------------------ ChatState ------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str  # persist thread title

# ------------------------ LLM Node ------------------------
llm = ChatOpenAI(max_retries=2)

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response], 'title': state['title']}  # persist title in state

# ------------------------ SQLite Checkpointer ------------------------
connect = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connect)


# ------------------------ Tools ------------------------
@tool
def search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo and return the top results.

    Args:
        query (str): The search query string.

    Returns:
        str: A formatted string of the top search results including title, URL, and snippet.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='us-en', max_results=5))
        if not results:
            return "No results found for the given query."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def calculator(expression: str) -> dict:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression (str): A string containing a basic arithmetic expression 
                          (e.g., "396.73 * 50"). Supports +, -, *, /.

    Returns:
        dict: A dictionary with:
            - 'expression' (str): The expression that was evaluated.
            - 'result' (float): The calculated result.
            - 'error' (str, optional): Error message if the calculation failed.

    """
    try:
        result = float(eval(expression))
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. AAPL, TSLA)
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=CE8MY894ND1ESUK5"
    
    r = requests.get(url).json()
    
    try:
        price = float(r["Global Quote"]["05. price"])
        return {"symbol": symbol, "price": price}
    except:
        return {"error": "Could not fetch stock price"}

async def load_mcp_tools():
    client = MultiServerMCPClient({
        'expense_tracker':{
            'command': r'C:\Users\lamic\Desktop\Expense MCP Server\.venv\Scripts\python.exe',
            'args': [r'C:\Users\lamic\Desktop\Expense MCP Server\main.py'],
            'transport':'stdio',
        }
    })
    return await client.get_tools()

mcp_tools = asyncio.run(load_mcp_tools())
#Making tool list
tools=[get_stock_price, search_tool, calculator]+mcp_tools

# Make the LLM tool_aware
llm_with_tools=llm.bind_tools(tools)

def chat_node(state:ChatState):
    '''
    LLM node that may answer or request a tool call.
    '''
    messages=state['messages']
    response=llm_with_tools.invoke(messages)
    return {'messages':[response]}


# Custom ToolNode that unwraps MCP's [{type, text, id}] format into plain text
from langchain_core.messages import ToolMessage

import asyncio
import json

async def _run_tool(tool, args):
    """Run a tool — async if supported, sync fallback otherwise."""
    try:
        raw = await tool.ainvoke(args)
    except NotImplementedError:
        raw = tool.invoke(args)
    # Unwrap MCP list format: [{type, text, id}] -> plain text
    if isinstance(raw, list):
        return ' '.join(
            block.get('text', str(block))
            for block in raw
            if isinstance(block, dict)
        )
    elif isinstance(raw, dict):
        return json.dumps(raw)
    return str(raw)

def clean_tool_node(state: ChatState):
    last_message = state['messages'][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_call_id = tool_call['id']

        matched_tool = next((t for t in tools if t.name == tool_name), None)
        if matched_tool is None:
            result = f'Tool {tool_name} not found'
        else:
            try:
                result = asyncio.run(_run_tool(matched_tool, tool_args))
            except Exception as e:
                result = f'Error: {str(e)}'

        tool_results.append(ToolMessage(content=result, tool_call_id=tool_call_id))

    return {'messages': tool_results}

tool_node = clean_tool_node

# ------------------------ Graph ------------------------
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot=graph.compile()
chatbot

chatbot = graph.compile(checkpointer=checkpointer)

# Wrap stream with a recursion limit so tool failures stop gracefully
CHATBOT_CONFIG_DEFAULTS = {'recursion_limit': 10}

# ------------------------ Retrieve all threads ------------------------
def retreive_all_threads():
    """
    Returns all threads as {thread_id: title}, ordered oldest→newest.
    Uses only the LATEST checkpoint per thread to get the most up-to-date title.
    checkpointer.list() yields checkpoints newest-first, so the first time
    we see a thread_id is always the most recent checkpoint for that thread.
    We collect them in seen order, then reverse so oldest is first (index 0)
    and newest is last — matching the original sidebar display logic.
    """
    seen = {}   # thread_id -> title, insertion order = newest checkpoint first

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable'].get('thread_id')
        if thread_id in seen:
            # Already captured the latest checkpoint for this thread — skip older ones
            continue
        title = checkpoint.checkpoint.get('channel_values', {}).get('title', 'New Chat')
        seen[thread_id] = title

    # Reverse so the dict is oldest→newest (sidebar shows reversed list, newest at top)
    return dict(reversed(list(seen.items())))

# ------------------------ Delete a thread ------------------------
def delete_thread(thread_id):
    """
    Deletes all checkpoints associated with a thread_id from the SQLite DB.
    Dynamically checks which tables exist to handle different LangGraph versions.
    """
    # Get all table names in the DB that have a thread_id column
    tables = connect.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]

    for table in table_names:
        # Check if this table has a thread_id column before deleting
        columns = [col[1] for col in connect.execute(f"PRAGMA table_info({table})").fetchall()]
        if "thread_id" in columns:
            connect.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))

    connect.commit()