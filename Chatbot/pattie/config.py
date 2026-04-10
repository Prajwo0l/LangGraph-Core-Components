# pattie/config.py
# =============================================================================
# Central place for every hard-coded path, constant, and environment setup.
# Change things here — nowhere else needs to know.
# =============================================================================
from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Personal Chatbot"
warnings.filterwarnings("ignore", message="could not convert string to float")

# ── Directory roots ──────────────────────────────────────────────────────────
# Must match the BASE_DIR used by the MCP filesystem server.
BASE_DIR = Path(r"C:\Users\lamic\Downloads").resolve()

# ── MCP server launch configs ────────────────────────────────────────────────
MCP_SERVERS: dict = {
    "expense_tracker": {
        "command": r"C:\Users\lamic\Desktop\Expense MCP Server\.venv\Scripts\python.exe",
        "args":    [r"C:\Users\lamic\Desktop\Expense MCP Server\main.py"],
        "transport": "stdio",
        "cwd":     r"C:\Users\lamic\Desktop\Expense MCP Server",
    },
    "filesystem": {
        "command": r"C:\Users\lamic\Desktop\File-System-MCP-Server\.venv\Scripts\python.exe",
        "args":    [r"C:\Users\lamic\Desktop\File-System-MCP-Server\main.py"],
        "transport": "stdio",
        "cwd":     r"C:\Users\lamic\Desktop\File-System-MCP-Server",
    },
}

# ── Filesystem tool classification ──────────────────────────────────────────
# Tools that MUST get human approval before executing (destructive ops).
FS_HITL_TOOLS: set[str] = {"write_file", "write_multiple_files", "delete_file", "delete_folder"}

# Tools that are safe to run automatically (read-only ops).
FS_AUTO_TOOLS: set[str] = {"list_files", "read_file", "read_multiple_files"}

# ── LangGraph config defaults ────────────────────────────────────────────────
CHATBOT_CONFIG_DEFAULTS: dict = {"recursion_limit": 25}

# ── Expense tracker database path (used by the Streamlit dashboard) ──────────
EXPENSE_DB = r"C:\Users\lamic\Desktop\Expense MCP Server\expenses.db"
