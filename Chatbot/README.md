# ✨ Pattie — Personal AI Assistant

A production-grade personal AI assistant built with **LangGraph**, **Streamlit**, and **MCP (Model Context Protocol)**. Pattie features an intent-routing graph architecture, persistent multi-thread conversations, RAG over uploaded PDFs, short-term and long-term memory, real-time expense tracking with Google Calendar integration, filesystem management, and human-in-the-loop approval flows.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Memory System](#memory-system)
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
| 💬 **Persistent Multi-Thread Chat** | Conversations saved to SQLite via LangGraph checkpointer. Fully restored on browser refresh. |
| 🧠 **Intent Router** | Lightweight LLM classifier routes each message to the right tool group — reduces hallucinations and cost. |
| 📝 **Short-Term Memory (STM)** | After 10 messages, older turns are compressed into a rolling summary. The last 8 turns are always kept verbatim. Safe deduplication prevents OpenAI 400 errors. |
| 🧬 **Long-Term Memory (LTM)** | Atomic facts are extracted from every conversation turn and stored in a FAISS vector index on disk. Semantically relevant facts are injected into every new conversation — across all threads. |
| 👤 **Structured User Profile** | LTM builds and maintains a structured profile (name, location, occupation, preferences, goals, expertise) that improves over time. |
| 🔍 **Web Search** | DuckDuckGo search for current events and real-time information. |
| 🧮 **Safe Calculator** | Arithmetic expression evaluator with character whitelist — no arbitrary code execution. |
| 📈 **Stock Prices** | Live equity prices via Alpha Vantage API. |
| 💸 **Expense Tracking** | Add, list, edit, delete, and summarize expenses via a local MCP server backed by SQLite. |
| 📅 **Google Calendar Integration** | Expenses are optionally added to Google Calendar with a human-in-the-loop approval step. |
| ✋ **HITL — Calendar Approval** | After an expense is saved, Pattie shows an approval card. Calendar tool is called directly — bypassing the LLM — for guaranteed reliability. |
| 🗂️ **Filesystem Management** | List, read, write, and delete files in a sandboxed Downloads directory via a dedicated MCP server. |
| ✋ **HITL — File Write/Delete** | Any destructive filesystem operation (write, delete file, delete folder) pauses for explicit user approval before executing. |
| 📄 **RAG — PDF Chat** | Upload any PDF per chat thread. Pattie chunks, embeds, and retrieves relevant passages using FAISS + OpenAI embeddings. PDFs in the Downloads folder can also be opened directly. |
| 📊 **Expense Dashboard** | Monthly bar chart, category breakdown with percentages, and total spending metric. |
| ⬇️ **CSV Export** | Download a summary or full detail CSV for any month. |
| 🧠 **Memory Panel** | Dedicated UI tab to inspect and manage STM summaries and all LTM facts, with per-fact delete and full wipe options. |

---

## Architecture

### LangGraph Graph

```
START
  │
  ▼
intent_router         ← Fast LLM call. Classifies message into one of 6 intents.
  │                     Also short-circuits to 'document' if a PDF is already loaded.
  ▼
chat_node             ← Main LLM node. Applies STM compression + injects LTM facts.
  │                     Receives ONLY the tools relevant to the current intent.
  │
  ├── (rag_tool / expense / search / finance calls)
  │         └──→ tool_node ──→ chat_node  (loop until no more tool calls)
  │
  ├── (read_file on .pdf)
  │         └──→ pdf_ingest ──→ chat_node  (auto-ingest PDF, then answer via RAG)
  │
  ├── (write_file / delete_file / delete_folder)
  │         └──→ fs_hitl ──→ END  (pauses; frontend handles approval + resume)
  │
  └── (no tool calls)
            └──→ ltm_update ──→ END  (extract facts → FAISS + profile, then done)
```

### Intent → Tool Mapping

| Intent | Tools Available | Triggered By |
|---|---|---|
| `expense` | `add_expense`, `list_expenses`, `summarize`, `edit_expense`, `delete_expense`, `set_budget`, `list_budgets`, `check_budget_alerts`, `monthly_overview`, `add_credit`, and more | "add 500 for lunch", "show my expenses", "set a food budget" |
| `filesystem` | `list_files`, `read_file`, `write_file`, `delete_file`, `delete_folder` | "list my downloads", "write a file", "open report.pdf from Downloads" |
| `search` | `search_tool` | "search for...", "latest news about...", "what is X" |
| `document` | `rag_tool` | "summarize the PDF", "explain this document", "what does it say about..." |
| `finance` | `get_stock_price`, `calculator` | "AAPL price", "what is 250 * 12" |
| `general` | *(none)* | "hello", casual conversation |

### HITL — Calendar Flow

```
User: "Add 500 for Netflix"
  │
  ▼
add_expense tool → saved to expenses.db ✅
  │
  ▼
Frontend detects expense result → shows approval card
  │
  ├── ✅ Approve → add_to_calendar called DIRECTLY (bypasses LLM + router)
  │                → Google Calendar event created ✅
  └── ❌ Reject  → Expense saved only. Calendar skipped.
```

### HITL — Filesystem Flow

```
User: "Write a Python file to Downloads/hello.py"
  │
  ▼
chat_node calls write_file(path, content)
  │
  ▼
fs_hitl_node intercepts → stores pending call in state
  → injects placeholder ToolMessage (satisfies LangGraph tool_call_id requirement)
  → graph pauses at END
  │
  ▼
Frontend reads fs_hitl_pending → shows approval card with content preview
  │
  ├── ✅ Approve → execute_tool_call() runs pathlib write directly
  │                → chatbot.update_state() injects real ToolMessage
  │                → graph resumes streaming → LLM gives final response
  └── ❌ Reject  → "cancelled" ToolMessage injected → graph resumes → LLM acknowledges
```

---

## Memory System

Pattie has a two-layer memory system implemented in `memory.py`.

### Short-Term Memory (STM)

STM is **per-thread** and stored in `stm_store.db`.

- Every call to `chat_node` passes messages through `apply_stm()` before sending to OpenAI
- `apply_stm()` first **deduplicates** messages by `tool_call_id` to prevent OpenAI 400 errors from LangGraph checkpoint replays
- Messages are then grouped into **atomic turns** — an `AIMessage` with `tool_calls` always stays with its `ToolMessage` responses; they are never split
- If the real turn count exceeds **10**, the oldest turns are compressed into a rolling summary using a separate LLM call
- The **last 8 turns** are always kept verbatim
- The summary is prepended as a `SystemMessage` so context is never lost

```
Turn count ≤ 10  →  [existing summary?] + all messages
Turn count > 10  →  [new compressed summary] + last 8 turns (atomic groups intact)
```

### Long-Term Memory (LTM)

LTM is **cross-thread** and stored in `ltm_store/`.

After every final assistant response, `ltm_update_node` runs:

1. **Fact Extraction** — an LLM call extracts atomic facts (≤ 20 words each) about the user from the last exchange
2. **Deduplication** — facts already in the store are skipped
3. **FAISS Indexing** — new facts are embedded and added to the vector index (`ltm_store/faiss_index/`)
4. **Profile Update** — a second LLM call merges new facts into a structured profile (`ltm_meta.json`)

On every `chat_node` call, `build_memory_context()` does a semantic search over the FAISS index for the top 5 facts most relevant to the current message, and injects them + the structured profile into the system prompt.

```
LTM Storage layout:
  ltm_store/
    faiss_index/     ← FAISS vector index (facts as embeddings)
    ltm_meta.json    ← { facts: [...], profile: { name, location, occupation, ... } }

stm_store.db         ← SQLite: per-thread rolling summaries
```

### Memory Panel (UI)

The **🧠 Memory** tab in the app lets you:

- View the **current thread's STM summary**
- Browse STM summaries from all other threads
- Inspect the **structured user profile** (name, location, job, preferences, goals, expertise)
- Browse all **atomic facts** with source thread and date, with a filter box
- **Delete individual facts** with the 🗑️ button (FAISS index is rebuilt automatically)
- **Wipe all LTM** with the clear button

---

## Project Structure

```
LangGraph-Core-Components/
│
├── Chatbot/
│   ├── langraph_backend.py      # LangGraph graph, intent router, all nodes, tools, MCP client
│   ├── streamlit_frontend.py    # Streamlit UI — Chat, Expense Dashboard, Memory tabs
│   ├── memory.py                # STM + LTM engine (dedup, grouping, FAISS, profile)
│   ├── chatbot.db               # SQLite — LangGraph conversation checkpoints
│   ├── stm_store.db             # SQLite — per-thread STM rolling summaries
│   ├── ltm_store/
│   │   ├── faiss_index/         # FAISS vector index for LTM facts
│   │   └── ltm_meta.json        # All atomic facts + structured user profile
│   └── README.md                # This file
│
├── Expense MCP Server/
│   ├── main.py                  # FastMCP server — expense tools + Google Calendar
│   ├── expenses.db              # SQLite — expense records
│   ├── categories.json          # Expense category definitions
│   ├── credentials.json         # Google OAuth client secret  ← DO NOT COMMIT
│   └── token.json               # Google OAuth access token   ← DO NOT COMMIT
│
└── File-System-MCP-Server/
    └── main.py                  # FastMCP server — sandboxed file operations on Downloads
```

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key
- A Google Cloud project with Calendar API enabled (for expense → calendar feature)

### 1. Set up the Chatbot environment

```bash
cd "LangGraph-Core-Components"
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in `LangGraph-Core-Components/Chatbot/`:

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

### 4. Set up the Filesystem MCP Server

```bash
cd "File-System-MCP-Server"
python -m venv .venv
.venv\Scripts\activate

pip install fastmcp
```

### 5. Configure Google Calendar (one-time)

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

A browser window opens. Approve access. `token.json` handles auth silently after that.

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

Every user message first hits `intent_router`. It uses a fast, zero-temperature LLM call to classify the message into one of six intents. Before the LLM call, it checks whether a PDF is already loaded for the current thread — if so, and the message looks like a document question (contains "summarize", "explain", "pdf", "what does it say", etc.) without explicit filesystem triggers, it short-circuits directly to `document` intent. A post-LLM safety override catches any remaining misclassifications.

### STM — Safe Message Compression

`apply_stm()` in `memory.py` is called at the start of every `chat_node` execution. It does three things in order:

1. **Deduplicates** — scans reversed message list, drops any `ToolMessage` whose `tool_call_id` has already been seen (LangGraph checkpoints can accumulate duplicates across HITL resume cycles)
2. **Groups** — runs `_group_turns()` which walks the deduplicated list and groups each `(AIMessage + its ToolMessages)` into an atomic unit that is never split
3. **Compresses** — counts real Human/AI turns; if over the threshold, builds a rolling summary from the oldest turns and returns `[summary] + last 8 turns`

### LTM — Cross-Thread Fact Memory

After every final assistant response, the graph passes through `ltm_update_node`. It finds the last Human/AI exchange, calls `update_ltm()` which sends it to the LLM with a structured extraction prompt, and stores any new atomic facts. On the next chat turn, `build_memory_context()` retrieves the top 5 semantically relevant facts via FAISS and includes them in the system prompt — making Pattie aware of who you are from the very first message of a new thread.

### RAG (PDF Chat)

1. Upload a PDF via the `📎 Upload a PDF` expander
2. The PDF is loaded with `PyPDFLoader`, split into 1000-character chunks with 200-character overlap
3. Chunks are embedded with `text-embedding-3-small` and stored in a per-thread in-memory FAISS retriever
4. When intent is `document`, `rag_tool` retrieves the 4 most semantically similar chunks and returns them as context
5. Each chat thread has its own independent document — switching threads switches the active retriever
6. PDFs in the Downloads folder can also be opened via `filesystem` intent → `read_file` → auto-intercepted by `pdf_ingest_node`

### MCP Integration

Both the Expense Tracker and Filesystem servers run as subprocesses launched by `MultiServerMCPClient` using `stdio` transport. The `tool_node` handles both sync LangChain tools and async MCP tools transparently via `_invoke_tool`. HITL filesystem operations bypass MCP entirely at approval time — they use `pathlib` directly to avoid the dead subprocess issue.

### Expense Dashboard

Switch to the `📊 Expense Dashboard` tab. Select a year and month. Pattie queries `expenses.db` directly (bypassing the LLM) and renders a total metric, a bar chart by category, a percentage breakdown, and two CSV download buttons.

---

## Tool Reference

### Built-in Tools

| Tool | Intent | Description |
|---|---|---|
| `search_tool` | search | DuckDuckGo web search, returns top 5 results with titles, URLs, and snippets |
| `calculator` | finance | Safe arithmetic eval — only allows `0-9 + - * / . ( ) % **` |
| `get_stock_price` | finance | Live equity price via Alpha Vantage GLOBAL_QUOTE endpoint |
| `rag_tool` | document | FAISS similarity search over the thread's uploaded PDF (top 4 chunks) |

### MCP Tools — Expense Tracker

| Tool | Description |
|---|---|
| `add_expense` | Saves expense. Returns structured result for HITL calendar approval. |
| `add_to_calendar` | Creates Google Calendar event. Called directly by frontend on approval. |
| `list_expenses` | Returns expenses between two dates |
| `summarize` | Total spending grouped by category |
| `edit_expense` / `delete_expense` | Modify or remove existing records |
| `set_budget` / `list_budgets` / `check_budget_alerts` / `delete_budget` | Budget management |
| `add_credit` / `list_credits` / `edit_credit` / `delete_credit` | Income / credit tracking |
| `monthly_overview` | Full monthly financial summary |

### MCP Tools — Filesystem

| Tool | HITL? | Description |
|---|---|---|
| `list_files` | No | Browse the Downloads directory |
| `read_file` | No | Read a text file. PDFs are auto-intercepted for RAG ingestion. |
| `write_file` | ✅ Yes | Create or overwrite a file — requires user approval |
| `delete_file` | ✅ Yes | Permanently delete a file — requires user approval |
| `delete_folder` | ✅ Yes | Permanently delete a folder and all contents — requires user approval |

---

## Important Files

| File | Purpose | Notes |
|---|---|---|
| `langraph_backend.py` | Full LangGraph graph, all nodes, tools, MCP client | Core backend |
| `streamlit_frontend.py` | Streamlit app — Chat, Dashboard, Memory tabs | Core frontend |
| `memory.py` | STM + LTM engine | Safe dedup, turn grouping, FAISS, profile |
| `chatbot.db` | LangGraph conversation checkpoints | Delete to reset all chat history |
| `stm_store.db` | Per-thread STM rolling summaries | Delete to reset all summaries |
| `ltm_store/ltm_meta.json` | All atomic LTM facts + user profile | Delete to wipe long-term memory |
| `ltm_store/faiss_index/` | FAISS vector index for LTM | Rebuilt automatically after fact deletion |
| `expenses.db` | Expense records | Never delete unless intentional |
| `token.json` | Google OAuth access token | Keep safe. Never commit. |
| `credentials.json` | Google OAuth client secret | Keep safe. Never commit. |
| `.env` | API keys | Never commit. |

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

### Filesystem MCP Server (`.venv`)

```
fastmcp
```

---

## Notes

- **Alpha Vantage free tier** allows 25 API calls/day. Get your own key at [alphavantage.co](https://www.alphavantage.co) and replace it in `get_stock_price`.
- **LTM extraction** makes one extra LLM call per conversation turn. For personal use this is negligible. Disable by removing the `ltm_update` node from the graph if cost is a concern.
- **STM deduplication** is essential because LangGraph's `add_messages` reducer + HITL `update_state()` cycles can produce duplicate `tool_call_id`s in the checkpoint. `apply_stm()` cleans these before every LLM call.
- **FAISS index** is rebuilt from scratch whenever a fact is deleted. For large fact stores this may take a second.
- If the **MCP servers fail to start** (e.g. `.venv` not set up), Pattie still loads and works — those tool groups will just be empty. A warning is printed to the terminal.
- **Conversation history** is stored in `chatbot.db`. Each thread can be deleted individually from the sidebar. Deleting a thread does not delete its LTM facts (those are global by design).
- The **Memory tab** reflects live state — changes (deletions) take effect immediately and persist across restarts.
