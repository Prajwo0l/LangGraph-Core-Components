from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv

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

# ------------------------ Graph ------------------------
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# ------------------------ Retrieve all threads ------------------------
def retreive_all_threads():
    """
    Returns all threads as {thread_id: title}.
    Pulls 'title' from the checkpoint values for persistence.
    """
    thread_dict = {}
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable'].get('thread_id')
        title = checkpoint.checkpoint.get('channel_values', {}).get('title', 'New Chat')  # fallback to 'New Chat'
        thread_dict[thread_id] = title
    return thread_dict
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