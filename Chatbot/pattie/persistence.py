# pattie/persistence.py
# =============================================================================
# SQLite-backed LangGraph checkpointer + thread management helpers.
# =============================================================================
from __future__ import annotations

import json
import sqlite3
from typing import Dict

from langgraph.checkpoint.sqlite import SqliteSaver

# ── Shared SQLite connection ─────────────────────────────────────────────────
_db_conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=_db_conn)


# ── Thread helpers ───────────────────────────────────────────────────────────

def retreive_all_threads() -> Dict[str, str]:
    """
    Return {thread_id: title} for every saved thread, most-recent first.

    Uses a direct SQLite query as a fallback because some versions of
    langgraph-checkpoint-sqlite have a broken .list() serialiser.
    """
    seen: Dict[str, str] = {}

    # Try the standard LangGraph API first
    try:
        for cp in checkpointer.list(None):
            tid = cp.config["configurable"].get("thread_id")
            if tid and tid not in seen:
                seen[tid] = cp.checkpoint.get("channel_values", {}).get("title", "New Chat")
        if seen:
            return dict(reversed(list(seen.items())))
    except Exception:
        pass

    # Fallback: raw SQL
    try:
        rows = _db_conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC"
        ).fetchall()
        for (tid,) in rows:
            if tid in seen:
                continue
            title = "New Chat"
            try:
                blob_row = _db_conn.execute(
                    "SELECT checkpoint FROM checkpoints "
                    "WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (tid,),
                ).fetchone()
                if blob_row:
                    data = json.loads(blob_row[0])
                    title = data.get("channel_values", {}).get("title") or "New Chat"
            except Exception:
                pass
            seen[tid] = title
    except Exception:
        pass

    return dict(reversed(list(seen.items())))


def delete_thread(thread_id: str) -> None:
    """Delete all checkpoint rows that belong to the given thread."""
    tables = _db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    for (table,) in tables:
        cols = [r[1] for r in _db_conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if "thread_id" in cols:
            _db_conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
    _db_conn.commit()
