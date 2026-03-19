"""
hitl.py — Multi-Agent Orchestration Graph (With Human-in-the-Loop)

Architecture:
─────────────────────────────────────────────────────────────────
              ┌──────────────────────────────────┐
              │         SUPERVISOR AGENT         │
              │  (routes tasks to sub-agents)    │
              └───────────┬──────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌────────────┐ ┌─────────────────────────────┐
   │  RESEARCHER │ │  ANALYST   │ │    TRADER AGENT              │
   │  AGENT      │ │  AGENT     │ │                              │
   │ (fetches    │ │ (analysis  │ │  ⚠️  HITL CHECKPOINT         │
   │  data)      │ │ + sizing)  │ │  Human must approve trade    │
   └─────────────┘ └────────────┘ │  before order is placed.    │
                                  └─────────────────────────────┘

HITL Checkpoints in this file:
  1. TRADE APPROVAL  — Before any purchase executes, the graph
                       pauses and waits for human "yes" / "no".
  2. RISK OVERRIDE   — If analyst flags HIGH risk, a second
                       interrupt asks the human to confirm anyway.

Flow:
  User → Supervisor → Researcher → Analyst → [HITL PAUSE] → Trader

Key Concepts Demonstrated:
  - interrupt() for mid-graph human checkpoints
  - Command(resume=...) to continue after approval
  - Multiple HITL gates (approval + risk override)
  - Graceful cancellation path if human says "no"
  - Full multi-agent orchestration (same as no_hitl.py)
  - MemorySaver persisting state across interrupt/resume cycle
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
import requests

load_dotenv()


# ─── 1. LLMs ────────────────────────────────────────────────────────────────
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
researcher_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
analyst_llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
trader_llm     = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─── 2. Tools ────────────────────────────────────────────────────────────────

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest real-time stock price for a given ticker symbol.
    Example symbols: 'AAPL', 'TSLA', 'GOOGL', 'MSFT'
    Returns: price, daily change, change percent, volume.
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
        "symbol":         quote.get("01. symbol"),
        "price":          quote.get("05. price"),
        "change":         quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
        "volume":         quote.get("06. volume"),
    }


@tool
def get_company_overview(symbol: str) -> dict:
    """
    Fetch fundamental company data: sector, PE ratio, market cap,
    52-week high/low, brief business description.
    Use this for deeper research before recommending a trade.
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
        "symbol":       data.get("Symbol"),
        "name":         data.get("Name"),
        "sector":       data.get("Sector"),
        "industry":     data.get("Industry"),
        "market_cap":   data.get("MarketCapitalization"),
        "pe_ratio":     data.get("PERatio"),
        "52_week_high": data.get("52WeekHigh"),
        "52_week_low":  data.get("52WeekLow"),
        "description":  data.get("Description", "")[:300] + "...",
    }


@tool
def calculate_position_size(price: float, budget: float, risk_percent: float = 2.0) -> dict:
    """
    Calculate optimal position size based on budget and risk tolerance.
    - price        : current stock price (USD)
    - budget       : total available capital (USD)
    - risk_percent : % of budget to risk (default 2%)
    Returns suggested_quantity, total_cost, risk_amount, risk_label.
    Risk labels: LOW (<2%), MEDIUM (2–5%), HIGH (>5%)
    """
    risk_amount = budget * (risk_percent / 100)
    quantity    = int(risk_amount / price)
    total_cost  = quantity * price

    # Risk classification — used by HITL gate to trigger extra warning
    if risk_percent < 2.0:
        risk_label = "LOW"
    elif risk_percent <= 5.0:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    return {
        "suggested_quantity": quantity,
        "risk_amount":        round(risk_amount, 2),
        "total_cost":         round(total_cost, 2),
        "risk_percent":       risk_percent,
        "risk_label":         risk_label,
    }


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Execute a mock stock purchase order.
    ⚠️  NOTE: In hitl.py this tool is only called AFTER human approval.
    Returns order confirmation with status FILLED.
    """
    return {
        "status":     "FILLED",
        "order_type": "MARKET",
        "symbol":     symbol,
        "quantity":   quantity,
        "message":    f"✅ Market order placed: {quantity} shares of {symbol} purchased.",
        "note":       "This is a simulation — no real funds were used.",
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
      messages         : Full conversation + tool result history.
      next             : Routing signal set by Supervisor ("researcher" /
                         "analyst" / "trader" / "finish").
      research_result  : Summary from Researcher agent.
      analysis_result  : Summary from Analyst agent.
      trade_result     : Confirmation from Trader agent.
      pending_trade    : Dict holding trade details waiting for HITL approval.
                         Format: {"symbol": str, "quantity": int, "risk_label": str}
      trade_approved   : True/False/None — result of the HITL approval gate.
    """
    messages:        Annotated[list[BaseMessage], add_messages]
    next:            str
    research_result: str
    analysis_result: str
    trade_result:    str
    pending_trade:   Optional[dict]   # Populated before HITL pause
    trade_approved:  Optional[bool]   # Set after human decision


# ─── 5. Supervisor Agent ──────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """You are a Stock Trading Supervisor coordinating a team of AI agents.

Your team:
  - researcher : Fetches stock prices and company fundamentals.
  - analyst    : Analyzes data and recommends position sizing.
  - trader     : Executes buy orders (requires human approval first).

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
    (only AFTER researcher + analyst have acted, unless user skips explicitly)
  - If trade_approved is False → FINISH (trade was cancelled by human)
  - If task is complete, greeting, or general chat → FINISH
"""

def supervisor_agent(state: AgentState) -> dict:
    """
    Supervisor node: orchestrates which sub-agent acts next.
    Reads completed work from state to make informed routing decisions.
    """
    # If human already rejected the trade, just finish
    if state.get("trade_approved") is False:
        print("\n🧠 Supervisor → Trade was rejected by human. Finishing.")
        return {"next": "finish"}

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

    valid_routes = {"researcher", "analyst", "trader", "finish"}
    if decision not in valid_routes:
        decision = "finish"

    print(f"\n🧠 Supervisor → routing to: {decision.upper()}")
    return {"next": decision}


# ─── 6. Sub-Agent Nodes ───────────────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a Stock Researcher.
Fetch stock prices and company fundamentals using your tools.
Use get_stock_price and get_company_overview for every stock mentioned.
After gathering data, provide a clear, factual summary.
"""

def researcher_agent(state: AgentState) -> dict:
    """Fetches market data and company fundamentals."""
    print("🔍 Researcher agent working...")
    messages = [SystemMessage(content=RESEARCHER_SYSTEM), *state["messages"]]
    response = researcher_llm_with_tools.invoke(messages)
    return {
        "messages":        [response],
        "research_result": response.content or "Research complete.",
    }


ANALYST_SYSTEM = """You are a Stock Analyst.
Analyze the research findings and recommend a position size.
Use calculate_position_size to compute quantity and risk.
Clearly state the risk level (LOW / MEDIUM / HIGH) in your output.
"""

def analyst_agent(state: AgentState) -> dict:
    """Analyzes research and recommends position sizing."""
    print("📊 Analyst agent working...")
    context  = f"Research findings: {state.get('research_result', 'None yet.')}"
    messages = [
        SystemMessage(content=ANALYST_SYSTEM),
        *state["messages"],
        HumanMessage(content=context),
    ]
    response = analyst_llm_with_tools.invoke(messages)
    return {
        "messages":        [response],
        "analysis_result": response.content or "Analysis complete.",
    }


# ─── 7. HITL Gate Node ────────────────────────────────────────────────────────
def hitl_approval_gate(state: AgentState) -> dict:
    """
    ⚠️  HUMAN-IN-THE-LOOP CHECKPOINT

    This node pauses the graph BEFORE any trade is executed.
    
    Two possible interrupts:
      A) Standard approval — always triggered before a trade.
      B) Risk override    — triggered additionally if analyst flagged HIGH risk.

    How interrupt() works:
      - interrupt(message) immediately suspends the graph.
      - The message string is returned in result["__interrupt__"][0].value
      - The graph saves full state to the checkpointer (MemorySaver).
      - The caller resumes with: chatbot.invoke(Command(resume=decision), config=...)
      - Execution continues from THIS function with `decision` as the return value.
    """
    analysis = state.get("analysis_result", "")
    research = state.get("research_result", "")
    is_high_risk = "HIGH" in analysis.upper()

    # Extract trade details to show the human
    pending = state.get("pending_trade") or {}
    symbol   = pending.get("symbol", "UNKNOWN")
    quantity = pending.get("quantity", 0)

    # ── HITL Gate A: Risk override (only for HIGH risk trades) ──────────────
    if is_high_risk:
        risk_prompt = (
            f"⚠️  HIGH RISK WARNING\n"
            f"The analyst has flagged this trade as HIGH RISK.\n\n"
            f"Details:\n{analysis}\n\n"
            f"Do you want to proceed anyway? (yes/no)"
        )
        print(f"\n{'='*60}")
        print("🚨 HITL INTERRUPT — HIGH RISK TRADE DETECTED")
        print(f"{'='*60}")
        
        risk_decision = interrupt(risk_prompt)
        
        if not (isinstance(risk_decision, str) and risk_decision.strip().lower() == "yes"):
            print("❌ Human rejected HIGH RISK trade.")
            return {"trade_approved": False}

    # ── HITL Gate B: Standard trade approval ────────────────────────────────
    approval_prompt = (
        f"📋 TRADE APPROVAL REQUEST\n"
        f"─────────────────────────\n"
        f"Stock   : {symbol}\n"
        f"Quantity: {quantity} shares\n\n"
        f"Research summary:\n{research[:200]}...\n\n"
        f"Analyst recommendation:\n{analysis[:300]}...\n\n"
        f"Do you approve this trade? (yes/no)"
    )
    print(f"\n{'='*60}")
    print("⏸️  HITL INTERRUPT — TRADE APPROVAL REQUIRED")
    print(f"{'='*60}")

    approval = interrupt(approval_prompt)

    if isinstance(approval, str) and approval.strip().lower() == "yes":
        print("✅ Human approved the trade.")
        return {"trade_approved": True}
    else:
        print("❌ Human rejected the trade.")
        return {"trade_approved": False}


# ─── 8. Trader Agent ──────────────────────────────────────────────────────────
TRADER_SYSTEM = """You are a Stock Trader.
Execute stock purchase orders using the purchase_stock tool.
You have already received human approval — proceed with confidence.
Report the order confirmation clearly.
"""

def trader_agent(state: AgentState) -> dict:
    """
    Trader node: executes the purchase ONLY if trade_approved is True.
    This node is only reached after hitl_approval_gate passes.
    """
    # Safety check — should always be True here, but guard anyway
    if not state.get("trade_approved"):
        return {
            "trade_result": "Trade was not approved. Order cancelled.",
            "messages": [AIMessage(content="Trade was cancelled — human did not approve.")]
        }

    print("💹 Trader agent executing approved order...")
    context = (
        f"Research: {state.get('research_result', 'N/A')}\n"
        f"Analysis: {state.get('analysis_result', 'N/A')}"
    )
    messages = [
        SystemMessage(content=TRADER_SYSTEM),
        *state["messages"],
        HumanMessage(content=f"Human has APPROVED this trade. Execute now.\n\n{context}"),
    ]
    response = trader_llm_with_tools.invoke(messages)
    return {
        "messages":    [response],
        "trade_result": response.content or "Trade executed.",
    }


# ─── 9. Tool Executor Nodes ───────────────────────────────────────────────────
researcher_tool_node = ToolNode(researcher_tools)
analyst_tool_node    = ToolNode(analyst_tools)
trader_tool_node     = ToolNode(trader_tools)


# ─── 10. Routing Functions ────────────────────────────────────────────────────

def supervisor_router(state: AgentState) -> Literal["researcher", "analyst", "trader_gate", "__end__"]:
    """
    Routes from supervisor to sub-agents.
    Note: 'trader' routes to 'trader_gate' (HITL), not directly to trader_agent.
    """
    next_node = state.get("next", "finish")
    if next_node == "finish":
        return END
    if next_node == "trader":
        return "trader_gate"   # ← Always go through HITL before trading
    return next_node


def researcher_router(state: AgentState) -> Literal["researcher_tools", "supervisor"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "researcher_tools"
    return "supervisor"


def analyst_router(state: AgentState) -> Literal["analyst_tools", "supervisor"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "analyst_tools"
    return "supervisor"


def after_hitl_router(state: AgentState) -> Literal["trader", "supervisor"]:
    """
    After the HITL gate:
    - Approved  → trader_agent executes
    - Rejected  → supervisor (which will then FINISH)
    """
    if state.get("trade_approved") is True:
        return "trader"
    return "supervisor"


def trader_router(state: AgentState) -> Literal["trader_tools", "supervisor"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "trader_tools"
    return "supervisor"


# ─── 11. Build the Graph ──────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Add all nodes ──
    graph.add_node("supervisor",       supervisor_agent)
    graph.add_node("researcher",       researcher_agent)
    graph.add_node("researcher_tools", researcher_tool_node)
    graph.add_node("analyst",          analyst_agent)
    graph.add_node("analyst_tools",    analyst_tool_node)
    graph.add_node("trader_gate",      hitl_approval_gate)   # ← HITL node
    graph.add_node("trader",           trader_agent)
    graph.add_node("trader_tools",     trader_tool_node)

    # ── Entry: always start at supervisor ──
    graph.add_edge(START, "supervisor")

    # ── Supervisor routes to sub-agents (or END) ──
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "researcher":  "researcher",
            "analyst":     "analyst",
            "trader_gate": "trader_gate",   # HITL gate sits between supervisor & trader
            END:           END,
        },
    )

    # ── Researcher loop ──
    graph.add_conditional_edges("researcher", researcher_router)
    graph.add_edge("researcher_tools", "researcher")

    # ── Analyst loop ──
    graph.add_conditional_edges("analyst", analyst_router)
    graph.add_edge("analyst_tools", "analyst")

    # ── HITL gate → trader (if approved) or supervisor (if rejected) ──
    graph.add_conditional_edges(
        "trader_gate",
        after_hitl_router,
        {
            "trader":     "trader",
            "supervisor": "supervisor",
        },
    )

    # ── Trader loop ──
    graph.add_conditional_edges("trader", trader_router)
    graph.add_edge("trader_tools", "trader")

    return graph


memory    = MemorySaver()
raw_graph = build_graph()
chatbot   = raw_graph.compile(checkpointer=memory, interrupt_before=["trader_gate"])
#                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  interrupt_before=["trader_gate"] tells LangGraph to always pause BEFORE
#  entering trader_gate, so the graph state is saved and we can resume cleanly.
#  The interrupt() calls inside hitl_approval_gate add additional granular pauses.


# ─── 12. CLI Runner ───────────────────────────────────────────────────────────

def handle_interrupts(result: dict, config: dict) -> dict:
    """
    Checks if the graph paused on an interrupt and handles it
    interactively in the CLI.
    Loops until all interrupts are resolved and graph finishes.
    """
    while True:
        interrupts = result.get("__interrupt__", [])
        if not interrupts:
            break

        # Display the interrupt prompt to the human
        prompt_text = interrupts[0].value
        print(f"\n{'='*60}")
        print(prompt_text)
        print(f"{'='*60}")

        human_decision = input("Your decision: ").strip()

        # Resume graph with the human's input
        result = chatbot.invoke(
            Command(resume=human_decision),
            config=config,
        )

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("📈 Stock Trading Multi-Agent System (WITH HITL)")
    print("=" * 60)
    print("Agents: Supervisor → Researcher / Analyst → [APPROVAL] → Trader")
    print("⚠️  All trades require your explicit approval before execution.")
    print("Type 'exit' to quit.\n")

    thread_id = "hitl-demo"

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break

        config: dict = {"configurable": {"thread_id": thread_id}}

        initial_state: AgentState = {
            "messages":        [HumanMessage(content=user_input)],
            "next":            "",
            "research_result": "",
            "analysis_result": "",
            "trade_result":    "",
            "pending_trade":   None,
            "trade_approved":  None,
        }

        # First graph run (may pause at trader_gate interrupt)
        result = chatbot.invoke(initial_state, config=config)

        # Handle any HITL interrupts until graph finishes
        result = handle_interrupts(result, config)

        # Find the last meaningful response
        final_message = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_message = msg.content
                break

        print(f"\nBot: {final_message or '(task complete — check agent logs above)'}\n")
        print("-" * 60)
