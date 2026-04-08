"""
orchestrator/graph.py — LangGraph-based pipeline orchestrator.

Pipeline nodes (LangGraph StateGraph):
  ┌──────────┐
  │  load    │  Load & clean the document (with optional page/chapter filter)
  └────┬─────┘
       │
  ┌────▼─────┐
  │  plan    │  Planner agent → sections
  └────┬─────┘
       │
  ┌────▼─────┐
  │  work    │  Worker agents run in parallel (asyncio)
  └────┬─────┘
       │
  ┌────▼──────┐
  │  review   │  Reviewer agent
  └────┬──────┘
       │
  ┌────▼──────┐
  │  write    │  Final writer agent
  └────┬──────┘
       │
  ┌────▼──────┐
  │  output   │  Format & return result
  └───────────┘
"""

import asyncio
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.planner_agent import run_planner
from agents.worker_agent import run_worker
from agents.reviewer_agent import run_reviewer
from agents.final_writer_agent import run_final_writer
from utils.document_loader import load_document
from utils.logger import get_logger
from config import Config

log = get_logger(__name__)


# ─── Shared pipeline state ────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # Inputs
    file_path: str
    cfg: Config
    page_range: Optional[tuple[int, int]]   # (start, end) 1-based, PDF only
    chapter: Optional[str]                  # named chapter / section to extract

    # Intermediate
    raw_text: str
    doc_title: str
    sections: list[dict]
    worker_outputs: list[dict]
    reviewed: dict

    # Final output
    result: dict
    error: str
    timing: dict[str, float]


# ─── Node functions ───────────────────────────────────────────────────────────

def node_load(state: PipelineState) -> PipelineState:
    """Load and clean the source document, applying page/chapter filters."""
    t0 = time.perf_counter()
    log.info("=== NODE: load ===")
    try:
        text = load_document(
            file_path=state["file_path"],
            page_range=state.get("page_range"),
            chapter=state.get("chapter"),
        )
        state["raw_text"] = text
    except Exception as exc:
        log.error("Load node failed: %s", exc)
        state["error"] = str(exc)
        state["raw_text"] = ""
    state.setdefault("timing", {})["load"] = round(time.perf_counter() - t0, 3)
    return state


def node_plan(state: PipelineState) -> PipelineState:
    """Run the planner agent to split the document into sections."""
    if state.get("error"):
        return state

    t0 = time.perf_counter()
    log.info("=== NODE: plan ===")
    try:
        plan = run_planner(state["raw_text"], state["cfg"])
        state["doc_title"] = plan["title"]
        state["sections"] = plan["sections"]
        log.info("Planner produced %d sections.", len(state["sections"]))
    except Exception as exc:
        log.error("Plan node failed: %s", exc)
        state["error"] = str(exc)
        state["sections"] = []
    state["timing"]["plan"] = round(time.perf_counter() - t0, 3)
    return state


def node_work(state: PipelineState) -> PipelineState:
    """Run worker agents concurrently using asyncio."""
    if state.get("error"):
        return state

    t0 = time.perf_counter()
    log.info("=== NODE: work (%d sections) ===", len(state["sections"]))

    cfg = state["cfg"]
    sections = state["sections"]

    async def run_all_workers() -> list[dict]:
        sem = asyncio.Semaphore(cfg.max_workers)

        async def bounded_worker(section: dict) -> dict:
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, run_worker, section, cfg)

        tasks = [bounded_worker(sec) for sec in sections]
        return await asyncio.gather(*tasks)

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    futures = [pool.submit(run_worker, sec, cfg) for sec in sections]
                    outputs = [f.result() for f in futures]
            else:
                outputs = loop.run_until_complete(run_all_workers())
        except RuntimeError:
            outputs = asyncio.run(run_all_workers())

        outputs.sort(key=lambda x: x.get("section_id", 0))
        state["worker_outputs"] = outputs
        log.info("Workers completed – %d summaries produced.", len(outputs))
    except Exception as exc:
        log.error("Work node failed: %s", exc)
        state["error"] = str(exc)
        state["worker_outputs"] = []

    state["timing"]["work"] = round(time.perf_counter() - t0, 3)
    return state


def node_review(state: PipelineState) -> PipelineState:
    """Run the reviewer agent."""
    if state.get("error"):
        return state

    t0 = time.perf_counter()
    log.info("=== NODE: review ===")
    try:
        reviewed = run_reviewer(
            doc_title=state.get("doc_title", "Untitled"),
            section_summaries=state["worker_outputs"],
            cfg=state["cfg"],
        )
        state["reviewed"] = reviewed
    except Exception as exc:
        log.error("Review node failed: %s", exc)
        state["error"] = str(exc)
        state["reviewed"] = {
            "reviewed_sections": state.get("worker_outputs", []),
            "global_notes": "",
        }
    state["timing"]["review"] = round(time.perf_counter() - t0, 3)
    return state


def node_write(state: PipelineState) -> PipelineState:
    """Run the final writer agent."""
    if state.get("error"):
        return state

    t0 = time.perf_counter()
    log.info("=== NODE: write ===")
    try:
        reviewed_sections = state["reviewed"].get("reviewed_sections", [])
        result = run_final_writer(
            doc_title=state.get("doc_title", "Untitled"),
            reviewed_sections=reviewed_sections,
            cfg=state["cfg"],
        )
        state["result"] = result
    except Exception as exc:
        log.error("Write node failed: %s", exc)
        state["error"] = str(exc)
        state["result"] = {}
    state["timing"]["write"] = round(time.perf_counter() - t0, 3)
    return state


def node_output(state: PipelineState) -> PipelineState:
    """Attach metadata and timing to the final result."""
    log.info("=== NODE: output ===")
    total = sum(state.get("timing", {}).values())
    state["timing"]["total"] = round(total, 3)

    result = state.get("result", {})
    result["_meta"] = {
        "file": state.get("file_path", ""),
        "page_range": state.get("page_range"),
        "chapter": state.get("chapter"),
        "timing_seconds": state.get("timing", {}),
        "error": state.get("error", None),
        "num_sections": len(state.get("sections", [])),
    }
    state["result"] = result

    log.info(
        "Pipeline complete. Total time: %.2fs | Sections: %d",
        total,
        len(state.get("sections", [])),
    )
    return state


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph pipeline."""
    g = StateGraph(PipelineState)

    g.add_node("load",   node_load)
    g.add_node("plan",   node_plan)
    g.add_node("work",   node_work)
    g.add_node("review", node_review)
    g.add_node("write",  node_write)
    g.add_node("output", node_output)

    g.set_entry_point("load")
    g.add_edge("load",   "plan")
    g.add_edge("plan",   "work")
    g.add_edge("work",   "review")
    g.add_edge("review", "write")
    g.add_edge("write",  "output")
    g.add_edge("output", END)

    return g.compile()


# ─── Convenience runner ───────────────────────────────────────────────────────

def run_pipeline(
    file_path: str,
    cfg: Config | None = None,
    page_range: tuple[int, int] | None = None,
    chapter: str | None = None,
) -> dict[str, Any]:
    """
    Run the full summarization pipeline on *file_path*.

    Parameters
    ----------
    file_path  : Path to a .pdf or .txt document.
    cfg        : Config instance; defaults to DEFAULT_CONFIG if None.
    page_range : Extract only these pages (1-based inclusive). PDF only.
                 E.g. (3, 7)  →  pages 3, 4, 5, 6, 7.
    chapter    : Extract only this named chapter/section.
                 Works for both PDF and TXT.
                 E.g. "Chapter 3" or "Introduction".

    Returns
    -------
    Final result dict with keys: title, section_summaries, final_summary, _meta.
    """
    from config import DEFAULT_CONFIG
    cfg = cfg or DEFAULT_CONFIG

    graph = build_graph()
    initial_state: PipelineState = {
        "file_path": file_path,
        "cfg": cfg,
        "page_range": page_range,
        "chapter": chapter,
        "timing": {},
    }

    final_state = graph.invoke(initial_state)

    if final_state.get("error"):
        log.error("Pipeline finished with error: %s", final_state["error"])

    return final_state.get("result", {"error": final_state.get("error", "Unknown error")})
