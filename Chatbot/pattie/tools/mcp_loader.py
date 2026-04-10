# pattie/tools/mcp_loader.py
# =============================================================================
# Loads tools from all configured MCP servers at startup.
# Isolated here so it can be mocked in tests and replaced without touching
# anything else.
# =============================================================================
from __future__ import annotations

import asyncio
from typing import List

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import MCP_SERVERS


async def _load_mcp_tools_async() -> list:
    client = MultiServerMCPClient(MCP_SERVERS)
    return await client.get_tools()


def load_mcp_tools() -> List:
    """
    Synchronously loads all MCP tools defined in config.MCP_SERVERS.
    Returns an empty list (with a warning) if any server fails to start.
    """
    try:
        return asyncio.run(_load_mcp_tools_async())
    except Exception as exc:
        print(f"[WARNING] Could not load MCP tools: {exc}")
        return []
