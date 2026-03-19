"""
no_hitl.py — Multi-Agent Orchestration Graph (No Human-in-the-Loop)

Architecture:
─────────────────────────────────────────────────────────────────
              ┌──────────────────────────────────┐
              │         SUPERVISOR AGENT         │
              │  (routes tasks to sub-agents)    │
              └───────────┬──────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌────────────┐ ┌─────────────────┐
   │  RESEARCHER │ │  ANALYST   │ │    TRADER        │
   │  AGENT      │ │  AGENT     │ │    AGENT         │
   │ (web-style  │ │ (price     │ │ (auto-executes   │
   │  stock      │ │  analysis) │ │  buy orders)     │
   │  lookup)    │ └────────────┘ └─────────────────┘
   └─────────────┘

Flow:
  User → Supervisor → (routes to) Researcher / Analyst / Trader
                   ↑_____________________________________________|
  Each agent loops back to Supervisor after completing its task.
  Supervisor decides when final answer is ready.

Key Concepts Demonstrated:
  - Multi-agent orchestration with a Supervisor node
  - Conditional routing (supervisor_router)
  - Sub-agents with dedicated tool sets
  - Shared state across all agents (AgentState)
  - MemorySaver for conversation persistence
  - Agent-to-agent handoff via "next" field in state
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
import requests

load_dotenv()


# ─── 1. LLMs ────────────────────────────────────────────────────────────────
# Supervisor uses a capable model for routing decisions
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Sub-agent LLMs (can be swapped for different models per agent)
researcher_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
analyst_llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
trader_llm     = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─── 2. Tools ────────────────────────────────────────────────────────────────

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest real-time stock price for a given ticker symbol.
    Example symbols: 'AAPL', 'TSLA', 'GOOGL', 'MSFT'
    Returns the Global Quote from Alpha Vantage.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    data = r.json()
    quote = data.get("Global Quote", {})
    if not quote:
        return {"error": f"No data found for symbol: {symbol}"}
    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
        "volume": quote.get("06. volume"),
    }


@tool
def get_company_overview(symbol: str) -> dict:
    """
    Fetch fundamental data about a company: sector, industry,
    market cap, PE ratio, 52-week high/low, description.
    Use this before making investment decisions.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=OVERVIEW&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    data = r.json()
    if not data or "Symbol" not in data:
        return {"error": f"No overview data found for symbol: {symbol}"}
    return {
        "symbol": data.get("Symbol"),
        "name": data.get("Name"),
        "sector": data.get("Sector"),
        "industry": data.get("Industry"),
        "market_cap": data.get("MarketCapitalization"),
        "pe_ratio": data.get("PERatio"),
        "52_week_high": data.get("52WeekHigh"),
        "52_week_low": data.get("52WeekLow"),
        "description": data.get("Description", "")[:300] + "...",
    }


@tool
def calculate_position_size(price: float, budget: float, risk_percent: float = 2.0) -> dict:
    """
    Calculate how many shares to buy based on budget and risk tolerance.
    - price: current stock price
    - budget: total available capital in USD
    - risk_percent: what % of budget to risk (default 2%)
    Returns suggested quantity and total cost.
    """
    risk_amount = budget * (risk_percent / 100)
    quantity = int(risk_amount / price)
    total_cost = quantity * price
    return {
        "suggested_quantity": quantity,
        "risk_amount": round(risk_amount, 2),
        "total_cost": round(total_cost, 2),
        "risk_percent": risk_percent,
    }


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Execute a mock stock purchase order.
    Simulates a brokerage order placement.
    Returns confirmation with order details.
    """
    return {
        "status": "FILLED",
        "order_type": "MARKET",
        "symbol": symbol,
        "quantity": quantity,
        "message": f"✅ Market order placed: {quantity} shares of {symbol} purchased.",
        "note": "This is a simulation — no real funds were used.",
    }


# ─── 3. Tool Groups per Agent ─────────────────────────────────────────────────
researcher_tools = [get_stock_price, get_company_overview]
analyst_tools    = [get_stock_price, calculate_position_size]
trader_tools     = [purchase_stock]

researcher_llm_with_tools = researcher_llm.bind_tools(researcher_tools)
analyst_llm_with_tools    = analyst_llm.bind_tools(analyst_tools)
trader_llm_with_tools     = trader_llm.bind_tools(trader_tools)


# ─── 4. Shared Agent State ────────────────────────────────────────────────────
class AgentState(TypedDict):
    """
    Shared state flowing through all nodes in the graph.

    Fields:
      messages : Full conversation + tool result history.
                 add_messages reducer appends new messages (no overwrite).
      next     : Which agent the Supervisor wants to invoke next.
                 "FINISH" signals the graph to end.
      research_result : Summary produced by the Researcher agent.
      analysis_result : Summary produced by the Analyst agent.
      trade_result    : Confirmation produced by the Trader agent.
    """
    messages:        Annotated[list[BaseMessage], add_messages]
    next:            str          # Routing signal set by Supervisor
    research_result: str          # Populated by researcher_agent
    analysis_result: str          # Populated by analyst_agent
    trade_result:    str          # Populated by trader_agent


# ─── 5. Supervisor Agent ──────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """You are a Stock Trading Supervisor coordinating a team of AI agents.

Your team:
  - researcher : Fetches stock prices and company fundamentals.
  - analyst    : Analyzes data and recommends position sizing.
  - trader     : Executes buy orders.

Your job:
  1. Read the user's request carefully.
  2. Decide which agent should act next.
  3. If all necessary work is done, respond with FINISH.

Always respond with ONLY one of these words (nothing else):
  researcher | analyst | trader | FINISH

Routing rules:
  - If the user wants stock info / company research → researcher
  - If the user wants analysis or how much to buy → analyst
  - If the user wants to buy / purchase a stock → trader
    (only after researcher + analyst have acted, unless user explicitly skips)
  - If the task is complete or just a greeting/chat → FINISH
"""

def supervisor_agent(state: AgentState) -> dict:
    """
    Supervisor node: reads conversation history + agent results,
    decides which sub-agent to call next (or FINISH).
    """
    # Build context for supervisor from completed work
    context_parts = []
    if state.get("research_result"):
        context_parts.append(f"[Research Done]: {state['research_result']}")
    if state.get("analysis_result"):
        context_parts.append(f"[Analysis Done]: {state['analysis_result']}")
    if state.get("trade_result"):
        context_parts.append(f"[Trade Done]: {state['trade_result']}")

    context = "\n".join(context_parts) if context_parts else "No sub-agent work done yet."

    supervisor_messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        *state["messages"],
        HumanMessage(content=f"Current completed work:\n{context}\n\nWho should act next?"),
    ]

    response = supervisor_llm.invoke(supervisor_messages)
    decision = response.content.strip().lower()

    # Normalize the decision
    valid_routes = {"researcher", "analyst", "trader", "finish"}
    if decision not in valid_routes:
        # Default: if we can't parse, go to FINISH to avoid infinite loop
        decision = "finish"

    print(f"\n🧠 Supervisor → routing to: {decision.upper()}")
    return {"next": decision}


# ─── 6. Sub-Agent Nodes ───────────────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a Stock Researcher.
Your job is to fetch stock prices and company fundamentals using your tools.
Always use get_stock_price and get_company_overview tools to gather data.
After collecting data, provide a clear summary of what you found.
"""

def researcher_agent(state: AgentState) -> dict:
    """
    Researcher node: fetches market data & company info.
    May call tools (get_stock_price, get_company_overview).
    """
    print("🔍 Researcher agent working...")
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        *state["messages"],
    ]
    response = researcher_llm_with_tools.invoke(messages)
    result_text = response.content or "Research complete."
    return {
        "messages": [response],
        "research_result": result_text,
    }


ANALYST_SYSTEM = """You are a Stock Analyst.
Your job is to analyze stock data and recommend position sizes using your tools.
Use calculate_position_size to determine how many shares to buy.
Provide a clear recommendation with reasoning.
"""

def analyst_agent(state: AgentState) -> dict:
    """
    Analyst node: evaluates the research and recommends position size.
    """
    print("📊 Analyst agent working...")
    context = f"Research findings: {state.get('research_result', 'None yet.')}"
    messages = [
        SystemMessage(content=ANALYST_SYSTEM),
        *state["messages"],
        HumanMessage(content=context),
    ]
    response = analyst_llm_with_tools.invoke(messages)
    result_text = response.content or "Analysis complete."
    return {
        "messages": [response],
        "analysis_result": result_text,
    }


TRADER_SYSTEM = """You are a Stock Trader.
Your job is to execute stock purchase orders using the purchase_stock tool.
Always confirm the symbol and quantity before placing the order.
Report the order confirmation clearly.
"""

def trader_agent(state: AgentState) -> dict:
    """
    Trader node: executes buy orders using the purchase_stock tool.
    No human approval — fully autonomous in no_hitl mode.
    """
    print("💹 Trader agent working...")
    context = (
        f"Research: {state.get('research_result', 'N/A')}\n"
        f"Analysis: {state.get('analysis_result', 'N/A')}"
    )
    messages = [
        SystemMessage(content=TRADER_SYSTEM),
        *state["messages"],
        HumanMessage(content=f"Execute the trade based on this context:\n{context}"),
    ]
    response = trader_llm_with_tools.invoke(messages)
    result_text = response.content or "Trade executed."
    return {
        "messages": [response],
        "trade_result": result_text,
    }


# ─── 7. Tool Executor Nodes ───────────────────────────────────────────────────
# Each agent gets its own ToolNode so tools are properly scoped

researcher_tool_node = ToolNode(researcher_tools)
analyst_tool_node    = ToolNode(analyst_tools)
trader_tool_node     = ToolNode(trader_tools)


# ─── 8. Routing Functions ─────────────────────────────────────────────────────

def supervisor_router(state: AgentState) -> Literal["researcher", "analyst", "trader", "__end__"]:
    """
    Reads the 'next' field set by supervisor_agent and routes accordingly.
    This is used by add_conditional_edges from the supervisor node.
    """
    next_node = state.get("next", "finish")
    if next_node == "finish":
        return END
    return next_node


def researcher_router(state: AgentState) -> Literal["researcher_tools", "supervisor"]:
    """
    After researcher_agent runs:
    - If the LLM wants to call tools → go to researcher_tools
    - Otherwise → return to supervisor for next routing decision
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "researcher_tools"
    return "supervisor"


def analyst_router(state: AgentState) -> Literal["analyst_tools", "supervisor"]:
    """After analyst_agent, route to tools or back to supervisor."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "analyst_tools"
    return "supervisor"


def trader_router(state: AgentState) -> Literal["trader_tools", "supervisor"]:
    """After trader_agent, route to tools or back to supervisor."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "trader_tools"
    return "supervisor"


# ─── 9. Build the Graph ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Add all nodes ──
    graph.add_node("supervisor",       supervisor_agent)
    graph.add_node("researcher",       researcher_agent)
    graph.add_node("researcher_tools", researcher_tool_node)
    graph.add_node("analyst",          analyst_agent)
    graph.add_node("analyst_tools",    analyst_tool_node)
    graph.add_node("trader",           trader_agent)
    graph.add_node("trader_tools",     trader_tool_node)

    # ── Entry point: always start at supervisor ──
    graph.add_edge(START, "supervisor")

    # ── Supervisor routes to sub-agents or END ──
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "researcher": "researcher",
            "analyst":    "analyst",
            "trader":     "trader",
            END:          END,
        },
    )

    # ── Researcher loop: agent → (tools → agent)* → supervisor ──
    graph.add_conditional_edges("researcher", researcher_router)
    graph.add_edge("researcher_tools", "researcher")

    # ── Analyst loop: agent → (tools → agent)* → supervisor ──
    graph.add_conditional_edges("analyst", analyst_router)
    graph.add_edge("analyst_tools", "analyst")

    # ── Trader loop: agent → (tools → agent)* → supervisor ──
    graph.add_conditional_edges("trader", trader_router)
    graph.add_edge("trader_tools", "trader")

    return graph


memory   = MemorySaver()
raw_graph = build_graph()
chatbot  = raw_graph.compile(checkpointer=memory)


# ─── 10. CLI Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("📈 Stock Trading Multi-Agent System (No HITL)")
    print("=" * 60)
    print("Agents: Supervisor → Researcher / Analyst / Trader")
    print("All trades execute AUTOMATICALLY — no approval needed.")
    print("Type 'exit' to quit.\n")

    thread_id = "no-hitl-demo"

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break

        initial_state: AgentState = {
            "messages":        [HumanMessage(content=user_input)],
            "next":            "",
            "research_result": "",
            "analysis_result": "",
            "trade_result":    "",
        }

        result = chatbot.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Find the last meaningful AI message to display
        final_message = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_message = msg.content
                break

        print(f"\nBot: {final_message or '(task complete — check agent logs above)'}\n")
        print("-" * 60)
