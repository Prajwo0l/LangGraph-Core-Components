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

from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random



os.environ['LANGCHAIN_PROJECT']='Personal Chatbot'
load_dotenv()

# ------------------------ ChatState ------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str  # persist thread title

# ------------------------ LLM Node ------------------------
llm = ChatOpenAI()

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response], 'title': state['title']}  # persist title in state

# ------------------------ SQLite Checkpointer ------------------------
connect = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connect)


# ------------------------ Tools ------------------------
search_tool=DuckDuckGoSearchRun(region='us-en')

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
    

#Making tool list
tools=[get_stock_price,search_tool,calculator]

# Make the LLM tool_aware
llm_with_tools=llm.bind_tools(tools)

def chat_node(state:ChatState):
    '''
    LLM node that may answer or request a tool call.
    '''
    messages=state['messages']
    response=llm_with_tools.invoke(messages)
    return {'messages':[response]}


tool_node=ToolNode(tools)

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