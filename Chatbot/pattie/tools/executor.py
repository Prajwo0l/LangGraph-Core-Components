# pattie/tools/executor.py
# =============================================================================
# Tool invocation helpers.
#
# WHY we bypass MCP for filesystem HITL tools
# ─────────────────────────────────────────────
# MCP tools talk over a stdio subprocess opened inside asyncio.run() at
# startup. That event loop is gone when the frontend later calls
# execute_tool_call(). A second asyncio.run() opens a brand-new loop with
# no live MCP connection, so write_file/delete_file silently do nothing.
#
# Solution: for every filesystem write/delete we execute the operation
# DIRECTLY in Python using pathlib — same sandbox logic as the MCP server,
# no subprocess needed.
# =============================================================================
from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

from ..config import BASE_DIR


# ── Low-level async invoker (used by tool_node for normal tool calls) ─────────

async def invoke_tool_async(tool: Any, args: dict) -> str:
    """
    Invoke a LangChain tool (async-first, sync fallback).
    Unwraps MCP content-block lists into plain strings.
    """
    try:
        raw = await tool.ainvoke(args)
    except NotImplementedError:
        raw = tool.invoke(args)

    if isinstance(raw, list):
        return " ".join(b.get("text", str(b)) for b in raw if isinstance(b, dict))
    if isinstance(raw, dict):
        return json.dumps(raw)
    return str(raw)


# ── Public helper called by the frontend after HITL approval ─────────────────

def execute_tool_call(tool_name: str, args: dict) -> str:
    """
    Execute a single tool by name.

    For filesystem write/delete tools we bypass the MCP subprocess and run
    the operation directly via pathlib (see module docstring).
    For all other tools we delegate to invoke_tool_async via asyncio.run().
    """

    # ── write_file ────────────────────────────────────────────────────────────
    if tool_name == "write_file":
        path_arg = args.get("path", "")
        content  = args.get("content", "")
        try:
            p = (BASE_DIR / path_arg).resolve()
            if not str(p).startswith(str(BASE_DIR)):
                return f"Access denied: '{path_arg}' is outside the Downloads directory."
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"File written successfully: {p}"
        except Exception as exc:
            return f"write_file error: {exc}"

    # ── write_multiple_files ──────────────────────────────────────────────────
    if tool_name == "write_multiple_files":
        files: dict = args.get("files", {})
        results: dict = {}
        for path_arg, content in files.items():
            try:
                p = (BASE_DIR / path_arg).resolve()
                if not str(p).startswith(str(BASE_DIR)):
                    results[path_arg] = f"Access denied: outside Downloads directory."
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(content), encoding="utf-8")
                results[path_arg] = f"Written: {p}"
            except Exception as exc:
                results[path_arg] = f"Error: {exc}"
        return json.dumps(results)

    # ── delete_file ───────────────────────────────────────────────────────────
    if tool_name == "delete_file":
        path_arg = args.get("path", "")
        try:
            p = (BASE_DIR / path_arg).resolve()
            if not str(p).startswith(str(BASE_DIR)):
                return f"Access denied: '{path_arg}' is outside the Downloads directory."
            if not p.exists():
                return f"File does not exist: {path_arg}"
            if not p.is_file():
                return f"'{path_arg}' is a directory — use delete_folder to remove directories."
            p.unlink()
            return f"File deleted: {path_arg}"
        except Exception as exc:
            return f"delete_file error: {exc}"

    # ── delete_folder ─────────────────────────────────────────────────────────
    if tool_name == "delete_folder":
        path_arg = args.get("path", "")
        try:
            p = (BASE_DIR / path_arg).resolve()
            if not str(p).startswith(str(BASE_DIR)):
                return f"Access denied: '{path_arg}' is outside the Downloads directory."
            if not p.exists():
                return f"Folder does not exist: {path_arg}"
            if not p.is_dir():
                return f"'{path_arg}' is a file — use delete_file to remove files."
            shutil.rmtree(p)
            return f"Folder deleted: {path_arg}"
        except Exception as exc:
            return f"delete_folder error: {exc}"

    # ── Fallback: any non-filesystem tool (e.g. calculator, search) ───────────
    from .registry import tools_by_name
    tool_obj = tools_by_name.get(tool_name)
    if tool_obj is None:
        return json.dumps({"error": f"Tool '{tool_name}' not found."})
    try:
        return asyncio.run(invoke_tool_async(tool_obj, args))
    except Exception as exc:
        return json.dumps({"error": str(exc)})
