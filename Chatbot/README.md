# ✨ Pattie — Personal AI Assistant

A production-grade personal AI assistant built with **LangGraph**, **Streamlit**, and **MCP (Model Context Protocol)**. Pattie features an intent-routing graph architecture, persistent multi-thread conversations, RAG over uploaded PDFs, real-time expense tracking with Google Calendar integration, and a human-in-the-loop approval flow.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Tool Reference](#tool-reference)
- [Important Files](#important-files)
- [Dependencies](#dependencies)

---

## Features

| Feature | Description |
|---|---|
| 💬 **Persistent Multi-Thread Chat** | Conversations saved to SQLite via LangGraph checkpointer. Restored on browser refresh. |
| 🧠 **Intent Router** | Lightweight LLM classifier routes each message to the right tool group before invoking the main model — faster and cheaper. |
| 🔍 **Web Search** | DuckDuckGo search for current events and real-time information. |
| 🧮 **Safe Calculator** | Arithmetic expression evaluator with character whitelist — no arbitrary code execution. |
| 📈 **Stock Prices** | Live equity prices via Alpha Vantage API. |
| 💸 **Expense Tracking** | Add, list, and summarize expenses via a local MCP server backed by SQLite. |
| 📅 **Google Calendar Integration** | Expenses are optionally added to Google Calendar with a human-in-the-loop approval step. |
| ✋ **HITL (Human-in-the-Loop)** | After every expense is saved, Pattie asks for explicit approval before creating the calendar event. The tool is called directly — bypassing the LLM — for guaranteed reliability. |
| 📄 **RAG — PDF Chat** | Upload any PDF per chat thread. Pattie chunks, embeds, and retrieves relevant passages using FAISS + OpenAI embeddings. |
| 📊 **Expense Dashboard** | Monthly bar chart, category breakdown with percentages, and total spending metric. |
| ⬇️ **CSV Export** | Download a summary or full detail CSV for any month. |

---

## Architecture

### LangGraph Graph

```
START
  │
  ▼
intent_router          ← Fast LLM call. Classifies the message into one of 5 intents.
  │                      Returns: expense | search | document | finance | general
  ▼
chat_node              ← Main LLM node. Receives ONLY the tools relevant to the intent.
  │                      Injects today's real date via system message.
  ├── (tool calls?) ──→ tool_node   ← Executes tools. Handles async MCP tools + sync tools.
  │                          │
  │                          └──→ chat_node  (loop until no more tool calls)
  │
  └── (no tool calls) ──→ END
```

### Intent → Tool Mapping

| Intent | Tools Available | Triggered By |
|---|---|---|
| `expense` | `add_expense`, `add_to_calendar`, `list_expenses`, `summarize` | "add 500 for lunch", "show my expenses" |
| `search` | `search_tool` | "what's the latest news about...", "search for..." |
| `document` | `rag_tool` | "what does the PDF say about...", "explain chapter 2" |
| `finance` | `get_stock_price`, `calculator` | "AAPL stock price", "what is 250 * 12" |
| `general` | *(none)* | "hello", "how are you", casual conversation |

### HITL Calendar Flow

```
User: "Add 500 for Netflix"
        │
        ▼
add_expense tool called
  → Saved to expenses.db ✅
  → Returns {status, date, amount, category, note}
        │
        ▼
Frontend detects add_expense result
  → Shows approval card in UI
        │
        ├── ✅ Approve → add_to_calendar called DIRECTLY (bypasses LLM + router)
        │                → Google Calendar event created ✅
        │
        └── ❌ Reject  → Expense saved only. Calendar skipped.
```

---

## Project Structure

```
LangGraph-Core-Components/
│
├── Chatbot/
│   ├── langraph_backend.py      # LangGraph graph, intent router, tools, RAG, MCP client
│   ├── streamlit_frontend.py    # Streamlit UI — chat tab + expense dashboard tab
│   ├── chatbot.db               # SQLite — LangGraph conversation checkpoints
│   └── README.md                # This file
│
└── Expense MCP Server/
    ├── main.py                  # FastMCP server — expense tools + Google Calendar
    ├── expenses.db              # SQLite — expense records
    ├── categories.json          # Expense category definitions
    ├── credentials.json         # Google OAuth client secret  ← DO NOT COMMIT
    └── token.json               # Google OAuth access token   ← DO NOT COMMIT
```

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key
- A Google Cloud project with Calendar API enabled (for expense → calendar feature)

---

### 1. Set up the Chatbot environment

```bash
cd "LangGraph-Core-Components"
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in `LangGraph-Core-Components/`:

```env
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls__...        # Optional — enables LangSmith tracing
LANGCHAIN_TRACING_V2=true        # Optional
```

### 3. Set up the Expense MCP Server

```bash
cd "Expense MCP Server"
python -m venv .venv
.venv\Scripts\activate

pip install fastmcp google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 4. Configure Google Calendar (one-time)

**Step 1** — Go to [Google Cloud Console](https://console.cloud.google.com)

**Step 2** — Enable the **Google Calendar API** for your project

**Step 3** — Create **OAuth 2.0 credentials** (Desktop app type) → Download as `credentials.json` → place in `Expense MCP Server/`

**Step 4** — Add your Gmail address as a test user:
`APIs & Services → OAuth consent screen → Audience → Add Users`

**Step 5** — Run the one-time auth flow to generate `token.json`:

```bash
cd "Expense MCP Server"
.venv\Scripts\python -c "
from google_auth_oauthlib.flow import InstalledAppFlow
SCOPES = ['https://www.googleapis.com/auth/calendar']
flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server(port=0)
with open('token.json', 'w') as f:
    f.write(creds.to_json())
print('Done — token.json saved.')
"
```

A browser window opens. Approve access. After that, `token.json` handles auth silently and never needs to be repeated unless deleted.

---

## Running the App

```bash
cd "LangGraph-Core-Components/Chatbot"
myenv\Scripts\activate
streamlit run streamlit_frontend.py
```

Open `http://localhost:8501` in your browser.

---

## How It Works

### Intent Router

Every user message first hits `intent_router` — a fast, zero-temperature LLM call that classifies the message into one of five intents. This node returns only the intent string; it calls no tools and does no generation. The result is stored in `ChatState.intent`.

`chat_node` then reads the intent and selects the pre-bound LLM variant that has only the relevant tools attached. This means:

- The LLM is never shown tools it doesn't need for the current task
- Tool selection hallucinations are drastically reduced
- Each intent gets a tailored system prompt hint

### RAG (PDF Chat)

1. Upload a PDF via the `📎 Upload a PDF` expander in the Chat tab
2. The PDF is loaded with `PyPDFLoader`, split into 1000-character chunks with 200-character overlap
3. Chunks are embedded with `text-embedding-3-small` and stored in a per-thread FAISS index
4. When you ask a question, `rag_tool` retrieves the 4 most semantically similar chunks and returns them as context
5. Each chat thread has its own independent document — switching threads switches the active retriever

### MCP Integration

The Expense MCP Server runs as a subprocess launched by `MultiServerMCPClient` using the `stdio` transport. The chatbot's `tool_node` handles both sync tools (search, calculator, stock) and async MCP tools transparently via `_invoke_tool`, which tries `ainvoke` first and falls back to `invoke`.

### Expense Dashboard

Switch to the `📊 Expense Dashboard` tab. Select a year and month from the dropdowns. Pattie queries `expenses.db` directly (not through the LLM or MCP) and renders:

- A **total spending metric**
- A **bar chart** grouped by category
- A **percentage breakdown table**
- Two **CSV download buttons** — one for the category summary, one for the full transaction detail

---

## Tool Reference

### Built-in Tools

| Tool | Intent Group | Description |
|---|---|---|
| `search_tool` | search | DuckDuckGo web search, returns top 5 results |
| `calculator` | finance | Safe arithmetic eval with character whitelist |
| `get_stock_price` | finance | Live equity price via Alpha Vantage |
| `rag_tool` | document | FAISS similarity search over uploaded PDF |

### MCP Tools (Expense Tracker)

| Tool | Intent Group | Description |
|---|---|---|
| `add_expense` | expense | Saves expense to SQLite. Returns expense details for HITL. |
| `add_to_calendar` | expense | Creates Google Calendar event. Called directly by frontend on HITL approval — never via LLM. |
| `list_expenses` | expense | Returns all expenses between two dates |
| `summarize` | expense | Returns total spending grouped by category |

---

## Important Files

| File | Purpose | Notes |
|---|---|---|
| `chatbot.db` | LangGraph conversation checkpoints | Delete to reset all chat history |
| `expenses.db` | Expense records | Never delete unless intentional |
| `token.json` | Google OAuth access token | Keep safe. Never commit to git. |
| `credentials.json` | Google OAuth client secret | Keep safe. Never commit to git. |
| `.env` | API keys | Never commit to git. |

---

## Dependencies

### Chatbot (`myenv`)

```
streamlit
langgraph
langchain
langchain-openai
langchain-community
langchain-mcp-adapters
faiss-cpu
pypdf
python-dotenv
ddgs
requests
```

### Expense MCP Server (`.venv`)

```
fastmcp
google-auth
google-auth-oauthlib
google-auth-httplib2
google-api-python-client
```

---

## Notes

- The **Alpha Vantage free tier** allows 25 API calls/day. Get your own key at [alphavantage.co](https://www.alphavantage.co) and replace it in `get_stock_price`.
- The **intent router** makes one extra LLM call per message. For a personal assistant this is negligible, but worth noting if you scale to many users.
- If the **MCP server fails to start** (e.g. `.venv` not set up), Pattie will still load and work — expense tools just won't be available. A warning is printed to the terminal.
- **Conversation history** is stored in `chatbot.db` as LangGraph checkpoints. Each chat thread is independent and can be deleted individually from the sidebar.
