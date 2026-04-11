# Document Summarizer

A production-grade, multi-agent document summarization pipeline built on **LangGraph**. Processes large PDF and TXT documents through a structured 6-node graph — planner, parallel workers, reviewer, and final writer — producing clean, structured summaries without RAG or vector databases.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Project structure](#project-structure)
3. [Quick start](#quick-start)
4. [CLI reference](#cli-reference)
5. [Agents](#agents)
6. [Utilities](#utilities)
7. [Configuration](#configuration)
8. [Output format](#output-format)
9. [LLM call budget](#llm-call-budget)
10. [Parallelism](#parallelism)
11. [Error handling](#error-handling)
12. [Running tests](#running-tests)
13. [Troubleshooting](#troubleshooting)

---

## How it works

The pipeline is a LangGraph `StateGraph` with six nodes sharing a single `PipelineState` TypedDict. Every node reads from and writes to this shared state — no data is passed between agents directly.

```
Document (.pdf / .txt)
        │
   ┌────▼─────┐
   │  load    │  Read file → apply page/chapter filter → clean text
   └────┬─────┘
        │
   ┌────▼─────┐
   │  plan    │  Planner agent → detect headings → slice sections  [1 LLM call]
   └────┬─────┘
        │
   ┌────▼─────┐
   │  work    │  Worker agents → summarize each section in parallel [N LLM calls]
   └────┬─────┘
        │
   ┌────▼──────┐
   │  review   │  Reviewer agent → deduplicate, unify terminology   [1 LLM call]
   └────┬──────┘
        │
   ┌────▼──────┐
   │  write    │  Final writer → synthesize one coherent summary    [1 LLM call]
   └────┬──────┘
        │
   ┌────▼──────┐
   │  output   │  Attach metadata + per-node timing → return result
   └───────────┘
```

**Total LLM calls = N + 3**, where N is the number of sections detected by the planner.

---

## Project structure

```
DocumentSummarizer/
│
├── main.py                     CLI entry point
├── config.py                   Central configuration dataclass
├── requirements.txt
│
├── agents/
│   ├── planner_agent.py        Splits document into sections         (1 LLM call)
│   ├── worker_agent.py         Summarizes one section, parallel-safe (1 LLM call)
│   ├── reviewer_agent.py       Refines all summaries as a batch      (1 LLM call)
│   └── final_writer_agent.py   Produces the final polished output    (1 LLM call)
│
├── orchestrator/
│   └── graph.py                LangGraph StateGraph — 6 nodes, full wiring
│
├── utils/
│   ├── document_loader.py      PDF/TXT loading, cleaning, page/chapter extraction
│   ├── chunker.py              Semantic + token-based smart chunking
│   ├── llm_client.py           OpenAI wrapper — retry, JSON repair, token counting
│   └── logger.py               Centralized console + file logging
│
├── tests/
│   └── test_summarizer.py      15 unit + integration tests (all LLM calls mocked)
│
└── sample_inputs/
    └── sample.txt              5-chapter sample document for quick testing
```

---

## Quick start

### 1. Activate the virtual environment

```powershell
& C:\Users\lamic\Desktop\LangGraph-Core-Components\myenv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Set up environment variables

The project reads `OPENAI_API_KEY` from the parent folder's `.env` automatically:

```
LangGraph-Core-Components/.env   ← add your key here
```

No separate `.env` inside `DocumentSummarizer/` is needed. Discovery order:
1. `DocumentSummarizer/.env` (local — checked first)
2. `LangGraph-Core-Components/.env` (parent — fallback)

The active `.env` path is printed at startup.

### 4. Run on the sample document

```powershell
cd "C:\Users\lamic\Desktop\LangGraph-Core-Components\DocumentSummarizer"
python main.py --file sample_inputs/sample.txt
```

---

## CLI reference

```
python main.py --file <path> [options]
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--file` | `-f` | required | Path to `.pdf` or `.txt` |
| `--chapter` | `-c` | — | Summarize only this chapter/section by name |
| `--pages` | `-p` | — | Page range, e.g. `10-25` (PDF only, 1-based) |
| `--depth` | `-d` | `medium` | `short` / `medium` / `detailed` |
| `--mode` | `-m` | `paragraph` | `bullet` / `paragraph` |
| `--model` | | `gpt-4o-mini` | Any OpenAI model name |
| `--workers` | `-w` | `4` | Max parallel worker threads |
| `--chunk-tokens` | | `1500` | Max tokens per document chunk |
| `--output` | `-o` | — | Save full JSON result to this path |
| `--plain` | | false | Print only the final summary text |
| `--log-level` | | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

`--chapter` and `--pages` are mutually exclusive.

**Examples:**

```powershell
# Full document, default settings
python main.py --file report.pdf

# Named chapter, detailed bullet summary
python main.py --file thesis.pdf --chapter "Literature Review" --depth detailed --mode bullet

# Page range, save output
python main.py --file book.pdf --pages 10-25 --output result.json

# Quiet output — summary text only
python main.py --file paper.pdf --chapter "Results" --plain
```

---

## Agents

### Planner agent

**File:** `agents/planner_agent.py` · **LLM calls:** 1

Analyses the full document and splits it into logical sections using a two-step approach designed to prevent hallucination:

1. The smart chunker splits the text into raw chunks (used as fallback).
2. The LLM is sent the full text and asked to return **structure only** — a list of verbatim heading strings and a document title. It never has to reproduce content.
3. Python slices the real text at the detected heading positions.

If the LLM call fails or returns no headings, the raw chunks from step 1 are used as sections.

Output schema:
```json
{
  "title": "Introduction to Machine Learning",
  "sections": [
    { "section_id": 0, "title": "Chapter 1", "text": "..." }
  ]
}
```

---

### Worker agent

**File:** `agents/worker_agent.py` · **LLM calls:** 1 per section (runs in parallel)

Summarizes a single section. Stateless and parallel-safe — designed to be submitted to a `ThreadPoolExecutor`.

Uses tiktoken `encode/decode` for token-accurate input truncation (fixes the blunt `text[:8000]` character cap that produces incorrect token counts for non-Latin text or code).

Depth controls:

| Depth | Summary length | Key points |
|---|---|---|
| `short` | 2–3 sentences | ≤ 3 |
| `medium` | 4–6 sentences | 4–6 |
| `detailed` | 7–10 sentences | 6–10 |

Fallback: returns a stub with the error message; does not crash the pipeline.

---

### Reviewer agent

**File:** `agents/reviewer_agent.py` · **LLM calls:** 1 (or 1 + 1 merge pass for large docs)

Processes all section summaries in a single batch and instructs the LLM to:

1. Remove or merge redundant key points appearing in multiple sections.
2. Unify inconsistent terminology across sections.
3. Flag logical gaps or contradictions.
4. Preserve original meaning — no new information added.

For payloads exceeding 6 000 tokens, sections are automatically split into batches, each reviewed separately, then merged in a final LLM pass. If the merge payload itself is too large, batches are sorted and returned directly.

Fallback: returns the original worker outputs unchanged.

---

### Final writer agent

**File:** `agents/final_writer_agent.py` · **LLM calls:** 1

Receives the reviewed sections and synthesizes them into a single coherent final summary. Respects `output_mode` (paragraph or bullet) and `summary_depth` (target word count: short < 200, medium 300–500, detailed 600–1000).

Fallback: concatenates section summaries with headings as plain text.

---

## Utilities

### Document loader (`utils/document_loader.py`)

Loads and cleans PDF or TXT files.

**PDF parsing chain:**
1. PyMuPDF (`fitz`) — fast, handles most layouts.
2. pdfplumber — fallback if PyMuPDF returns empty text.
3. `RuntimeError` with install instructions if neither is available.

**TXT encoding cascade:** `utf-8` → `utf-8-sig` → `latin-1` → `cp1252`.

**Page range slicing** (PDF only): validates the range against the actual page count before extracting.

**Chapter extraction** (PDF + TXT): scans for headings using a regex that matches Markdown headings, `Chapter/Section/Part N`, ALL-CAPS lines, and Title-Case standalone lines. Returns text from the matched heading to the next heading. If the name is not found, raises a `RuntimeError` listing all detected headings.

**`clean_text()` pipeline:**
1. Unicode NFC normalization
2. Replace non-breaking / zero-width spaces
3. Strip control characters (preserve `\n`, `\t`)
4. Normalize line endings
5. Strip trailing whitespace per line
6. Collapse 3+ blank lines → max 2
7. Re-join broken lines (line ends without punctuation + next starts lowercase)

---

### Chunker (`utils/chunker.py`)

Splits cleaned text into token-bounded chunks.

**Strategy (priority order):**
1. Semantic split on heading boundaries (regex).
2. Paragraph split on double newlines if no headings found.
3. For any chunk still over `max_chunk_tokens`: greedy paragraph packing, then token-boundary split with `overlap_tokens` overlap.
4. Merge small adjacent chunks — **only within the same semantic section** (guarded by a `section_tag` key to prevent boundary-crossing merges).

Token counting uses tiktoken `cl100k_base` with a graceful fallback for unknown model names.

---

### LLM client (`utils/llm_client.py`)

Single entry point for all LLM calls.

- Auto-loads `.env` at import time (local → parent folder).
- Exponential backoff retry on `RateLimitError`, `APITimeoutError`, `APIError`.
- **4-stage JSON repair** for large responses that contain raw newlines or malformed strings:
  1. Direct `json.loads` on stripped text.
  2. Strip markdown fences (` ```json ... ``` `).
  3. Extract the outermost `{ … }` block with a brace counter.
  4. `_repair_json()` — fixes raw newlines/tabs inside strings, trailing commas, control characters.

---

## Configuration

All parameters live in `config.py` as a single dataclass. CLI flags override defaults at runtime.

```python
@dataclass
class Config:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 4096

    max_chunk_tokens: int = 1500    # max tokens per chunk before splitting
    min_chunk_tokens: int = 100     # chunks below this get merged
    overlap_tokens: int = 50        # overlap between adjacent chunks

    summary_depth: Literal["short", "medium", "detailed"] = "medium"
    output_mode: Literal["bullet", "paragraph"] = "paragraph"

    max_workers: int = 4            # concurrent worker threads

    log_level: str = "INFO"
    log_file: str = "summarizer.log"
```

**Tuning guide:**

| Goal | Change |
|---|---|
| Fewer LLM calls (faster, cheaper) | Increase `--chunk-tokens` to 3000+ |
| More granular section summaries | Decrease `--chunk-tokens` to 800 |
| Higher quality output | Switch `--model gpt-4o` |
| Reduce rate limit errors | Decrease `--workers 2` |
| Very large document (100+ pages) | Increase `--workers 8` + `--chunk-tokens 2000` |

---

## Output format

### Console (default)

```
======================================================================
  DOCUMENT SUMMARY  [entire document]
  Title  : Introduction to Machine Learning
  File   : sample_inputs/sample.txt
  Time   : 90.07s  (load=0.00s  plan=21.37s  work=11.00s  review=32.00s  write=25.00s)
======================================================================

── SECTION SUMMARIES ──────────────────────────────────────────────

  [0] Introduction
  This document provides a comprehensive overview of machine learning ...

  [1] Chapter 1: What is Machine Learning?
  Machine learning is a branch of artificial intelligence ...

======================================================================
  FINAL SUMMARY
======================================================================
Machine learning (ML) is a transformative branch of artificial intelligence ...
```

### JSON (`--output result.json`)

```json
{
  "title": "Introduction to Machine Learning",
  "section_summaries": [
    { "section_id": 0, "title": "Introduction", "summary": "..." }
  ],
  "final_summary": "Machine learning is a transformative branch of AI...",
  "_meta": {
    "file": "sample_inputs/sample.txt",
    "page_range": null,
    "chapter": null,
    "num_sections": 7,
    "error": null,
    "timing_seconds": {
      "load": 0.003,
      "plan": 21.37,
      "work": 11.00,
      "review": 32.00,
      "write": 25.00,
      "total": 90.07
    }
  }
}
```

### Plain text (`--plain`)

Prints only the title and `final_summary` — no section breakdown or metadata.

---

## LLM call budget

```
Total = N + 3
      = N workers  (one per section — dynamic, scales with document size)
      + 1 planner  (structure detection only — fixed)
      + 1 reviewer (full batch — fixed)
      + 1 writer   (final synthesis — fixed)
```

| Scenario | Sections | Total calls |
|---|---|---|
| `sample.txt` — full document | 7 | 10 |
| `sample.txt` — single chapter | 1 | 4 |
| 100-page PDF — full | ~30 | ~33 |
| 100-page PDF — `--pages 1-10` | ~5 | ~8 |

---

## Parallelism

Workers use `concurrent.futures.ThreadPoolExecutor` with `as_completed()`.

All sections are submitted at once — up to `max_workers` threads run simultaneously. `as_completed()` collects results the instant each thread finishes rather than waiting on submission order.

`ThreadPoolExecutor` is used instead of `asyncio` because `run_worker()` is a plain synchronous blocking function (HTTP call + wait). `asyncio.gather()` only provides true concurrency for async coroutines; wrapping sync functions with `run_in_executor` still uses threads under the hood, so `ThreadPoolExecutor` is used directly.

**Example with 7 sections and 4 workers:**

```
t=0s  → Workers 0, 1, 2, 3 start simultaneously
t=4s  → Worker 1 finishes  → Worker 4 starts
t=4s  → Worker 3 finishes  → Worker 5 starts
t=6s  → Workers 0, 2 finish → Worker 6 starts
t=8s  → Workers 4, 5, 6 finish

Wall-clock ≈ 8s  (vs ~28s sequential)
```

Results are sorted by `section_id` after all threads finish to restore document order.

---

## Error handling

Every stage has a fallback — the pipeline always produces some output:

| Stage | Failure | Fallback |
|---|---|---|
| Load | File not found / parse error | Sets `error` flag; downstream nodes skip |
| Planner | LLM call fails | Uses raw chunker output as sections |
| Worker | LLM call fails | Returns stub with error message |
| Worker | Empty section text | Returns `"(Empty section)"` immediately |
| Reviewer | LLM call fails | Returns original worker outputs unchanged |
| Final writer | LLM call fails | Concatenates section summaries as plain text |
| LLM client | Rate limit / timeout | Exponential backoff, up to 3 retries |
| LLM client | Malformed JSON | 4-stage repair chain |
| PDF loader | PyMuPDF returns empty | Falls back to pdfplumber automatically |
| Chapter extract | Heading not found | Raises `RuntimeError` listing all available headings |

---

## Running tests

All 15 tests mock every LLM call — no API key required.

```powershell
pytest tests/ -v
pytest tests/ -v --tb=short
pytest tests/test_summarizer.py::TestPipelineIntegration -v
```

| Test class | Coverage |
|---|---|
| `TestCleanText` | Unicode normalization, blank-line collapse, control char removal |
| `TestChunker` | Chunk production, token limits, text preservation |
| `TestPlannerAgent` | Section detection, LLM fallback to chunks |
| `TestWorkerAgent` | Summary output, empty section, graceful LLM failure |
| `TestReviewerAgent` | Reviewed sections output, empty input |
| `TestFinalWriterAgent` | Final summary output, empty sections fallback |
| `TestPipelineIntegration` | Full end-to-end with mocked LLM; missing file error |

---

## Troubleshooting

**`OPENAI_API_KEY` not found**
Check that `LangGraph-Core-Components/.env` contains `OPENAI_API_KEY=sk-...` and confirm the startup log shows the correct `.env` path.

**PDF text is empty after parsing**
The PDF is likely scanned (image-only). Add OCR support with `pytesseract` + `pdf2image`.

**`ModuleNotFoundError: No module named 'fitz'`**
```powershell
pip install pymupdf
```

**`ModuleNotFoundError: No module named 'langgraph'`**
```powershell
pip install langgraph langchain-core
```

**Rate limit errors with many sections**
Reduce `--workers 2` to slow concurrent requests, or switch to a model tier with higher RPM limits.

**Chapter not found**
Run with `--log-level DEBUG` to see heading detection. Use the exact heading text listed in the error's "Available headings" output.

**`--pages` has no effect**
`--pages` is PDF-only. For `.txt` files, use `--chapter` instead.

**`--chapter` and `--pages` used together**
These are mutually exclusive — the CLI errors if both are provided. Use one or the other.

---

## Dependencies

```
openai>=1.30.0
langgraph>=0.2.0
langchain-core>=0.2.0
pymupdf>=1.24.0
pdfplumber>=0.11.0
tiktoken>=0.7.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI API key |
| `LANGCHAIN_TRACING_V2` | no | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | no | LangSmith API key |
| `LANGCHAIN_PROJECT` | no | LangSmith project name |

---

## Design constraints

- No RAG — no vector databases, no embeddings, no retrieval
- No external APIs except OpenAI (or any compatible endpoint)
- No monolithic code — each agent is an independent callable module
- No hardcoded values — everything flows through `Config`
- No sequential worker execution — all workers run in parallel via `ThreadPoolExecutor`

---

*MIT — free to use, modify, and distribute.*
