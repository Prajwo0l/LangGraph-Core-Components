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
  │  work    │  Worker agents run TRULY in parallel (ThreadPoolExecutor)
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

WHY ThreadPoolExecutor instead of asyncio
─────────────────────────────────────────
run_worker() is a plain synchronous blocking function (it calls the OpenAI
REST API and waits). For blocking I/O-bound work, threads are the correct
and simplest parallelism primitive:

  • asyncio.gather() only gives true concurrency for *async* coroutines.
    Wrapping a sync function with run_in_executor still uses threads under
    the hood — so we might as well use ThreadPoolExecutor directly.
  • ThreadPoolExecutor works identically in a plain script, inside a
    LangGraph node, and inside Jupyter — no event-loop juggling required.
  • concurrent.futures.as_completed() lets results stream in as they finish
    rather than blocking on each one in turn.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """
    Run all worker agents truly in parallel using ThreadPoolExecutor.

    How it works
    ────────────
    1. A ThreadPoolExecutor is created with max_workers threads.
    2. Every section gets its own future: pool.submit(run_worker, section, cfg)
    3. as_completed() yields each future the moment it finishes — workers are
       NOT waited on one-by-one; they all run at the same time (up to the
       thread limit) and results are collected as they arrive.
    4. Results are sorted by section_id at the end to restore document order.

    Example with 6 sections and max_workers=4
    ──────────────────────────────────────────
    t=0s  → sections 0,1,2,3 start simultaneously
    t=2s  → section 1 finishes first  → collected immediately
    t=3s  → section 0 finishes        → section 4 starts
    t=3s  → section 3 finishes        → section 5 starts
    t=4s  → sections 2,4,5 finish
    Total wall-clock ≈ 4s  (vs 12s+ sequential)
    """
    if state.get("error"):
        return state

    t0 = time.perf_counter()
    cfg = state["cfg"]
    sections = state["sections"]
    total = len(sections)

    log.info("=== NODE: work (%d sections, max_workers=%d) ===", total, cfg.max_workers)

    outputs: list[dict] = []

    try:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            # Submit ALL sections at once — they start running immediately
            future_to_section = {
                pool.submit(run_worker, section, cfg): section
                for section in sections
            }

            finished = 0
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                finished += 1
                try:
                    result = future.result()
                    outputs.append(result)
                    log.info(
                        "Worker[%d] ✓ finished (%d/%d)  '%s'",
                        section["section_id"],
                        finished,
                        total,
                        section.get("title", "")[:50],
                    )
                except Exception as exc:
                    # Individual worker failure — log and continue; don't crash pipeline
                    log.error(
                        "Worker[%d] ✗ failed (%d/%d): %s",
                        section["section_id"],
                        finished,
                        total,
                        exc,
                    )
                    outputs.append({
                        "section_id": section["section_id"],
                        "title": section.get("title", ""),
                        "key_points": [],
                        "summary": f"[Worker failed: {exc}]",
                    })

        # Restore document order (workers finish in arbitrary order)
        outputs.sort(key=lambda x: x.get("section_id", 0))
        state["worker_outputs"] = outputs
        elapsed = time.perf_counter() - t0
        log.info(
            "All workers done. %d/%d succeeded. Wall-clock: %.2fs",
            sum(1 for o in outputs if not o["summary"].startswith("[Worker failed")),
            total,
            elapsed,
        )

    except Exception as exc:
        log.error("node_work: unexpected error – %s", exc)
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
