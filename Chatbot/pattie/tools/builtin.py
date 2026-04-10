# pattie/tools/builtin.py
# =============================================================================
# Pure LangChain @tool definitions that do NOT require MCP.
# Each tool has a single, clearly-stated responsibility.
# =============================================================================
from __future__ import annotations

import requests
from ddgs import DDGS
from langchain_core.tools import tool

from ..rag import get_active_thread, get_retriever, thread_document_metadata


@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo. Use for current events or factual lookups."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="us-en", max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            for r in results
        )
    except Exception as exc:
        return f"Search error: {exc}"


@tool
def calculator(expression: str) -> dict:
    """
    Safely evaluate a basic arithmetic expression (+, -, *, /, **, %).
    Example: '396.73 * 50'  or  '(100 + 200) / 3'
    """
    allowed = set("0123456789+-*/.() %**\t\n")
    if not all(c in allowed for c in expression):
        return {"error": "Expression contains invalid characters."}
    try:
        result = float(eval(expression, {"__builtins__": {}}, {}))
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a ticker symbol (e.g. AAPL, TSLA, AMZN)."""
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=CE8MY894ND1ESUK5"
    )
    try:
        data = requests.get(url, timeout=10).json()
        price = float(data["Global Quote"]["05. price"])
        return {"symbol": symbol, "price": price}
    except Exception:
        return {"error": f"Could not fetch stock price for {symbol}."}


@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant passages from the PDF document currently loaded in this
    session (either uploaded by the user or opened from the filesystem).
    Use this whenever the user asks questions about a document or PDF.
    """
    thread_id = get_active_thread()
    retriever = get_retriever(thread_id)
    if retriever is None:
        return {
            "error": (
                "No document loaded yet. "
                "Ask the user to upload a PDF or open one from their Downloads folder."
            )
        }
    docs = retriever.invoke(query)
    return {
        "query": query,
        "context": [d.page_content for d in docs],
        "source_file": thread_document_metadata(thread_id).get("filename"),
    }
