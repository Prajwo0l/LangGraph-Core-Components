# 📄 Hierarchical Multi-Agent Document Summarization System

A **production-ready**, **LangGraph-powered** multi-agent pipeline that processes large PDF and TXT documents and generates high-quality, structured summaries — with zero RAG, zero vector databases, and full parallel execution.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🤖 Multi-Agent Pipeline | Planner → Workers → Reviewer → Final Writer |
| ⚡ Parallel Execution | Worker agents run concurrently via `asyncio` |
| 📐 Smart Chunking | Semantic boundaries (headings/paragraphs) + token-based fallback |
| 🔁 Hierarchical Summarization | Recursive split for oversized sections |
| 🎚️ Adaptive Depth | `short` / `medium` / `detailed` |
| 🖊️ Dual Output Modes | `bullet` points or flowing `paragraph` prose |
| 🔍 Redundancy Reduction | Reviewer detects and merges repeated ideas |
| ✅ Consistency Enforcement | Unified terminology across all sections |
| 🛡️ Error Handling | Graceful fallbacks at every stage |
| 📊 Logging & Timing | Per-node execution times, structured logs |
| ⚙️ Config System | All parameters in one place, CLI-overridable |

---

## 🗂️ Project Structure

```
DocumentSummarizer/
│
├── main.py                        ← CLI entry point
├── config.py                      ← Central configuration (all tunable params)
├── requirements.txt
├── .env.example                   ← Copy to .env and add your API key
│
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py           ← Splits document into logical sections
│   ├── worker_agent.py            ← Summarises one section (parallelizable)
│   ├── reviewer_agent.py          ← Consistency, redundancy, gap checks
│   └── final_writer_agent.py     ← Produces the clean final summary
│
├── orchestrator/
│   ├── __init__.py
│   └── graph.py                   ← LangGraph StateGraph pipeline
│
├── utils/
│   ├── __init__.py
│   ├── document_loader.py         ← PDF/TXT loading + text cleaning
│   ├── chunker.py                 ← Smart semantic + token-based chunker
│   ├── llm_client.py              ← OpenAI wrapper with retry logic
│   └── logger.py                  ← Centralized logging setup
│
├── tests/
│   ├── __init__.py
│   └── test_summarizer.py         ← Unit + integration tests (mocked LLM)
│
└── sample_inputs/
    └── sample.txt                 ← Ready-to-use sample document
```

---

## 🚀 Quick Start

### 1. Clone / Navigate to the project

```bash
cd "LangGraph-Core-Components/DocumentSummarizer"
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
# Copy the template
cp .env.example .env

# Edit .env and paste your key
OPENAI_API_KEY=sk-...
```

### 5. Run the summarizer

```bash
# Summarize the included sample document
python main.py --file sample_inputs/sample.txt

# Summarize a PDF
python main.py --file path/to/document.pdf

# Short bullet-point summary
python main.py --file report.pdf --depth short --mode bullet

# Detailed paragraph summary, saved to JSON
python main.py --file thesis.pdf --depth detailed --output result.json

# Print only the final summary (no metadata)
python main.py --file paper.pdf --plain
```

---

## ⚙️ All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--file` / `-f` | *(required)* | Path to `.pdf` or `.txt` file |
| `--depth` / `-d` | `medium` | `short` / `medium` / `detailed` |
| `--mode` / `-m` | `paragraph` | `bullet` / `paragraph` |
| `--model` | `gpt-4o-mini` | Any OpenAI model name |
| `--workers` / `-w` | `4` | Number of parallel worker agents |
| `--chunk-tokens` | `1500` | Max tokens per document chunk |
| `--output` / `-o` | None | Save full JSON result to file |
| `--plain` | False | Print only final summary text |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## 🏗️ Architecture Deep-Dive

### Pipeline Flow

```
Document (PDF/TXT)
        │
        ▼
┌───────────────┐
│  LOAD Node    │  PyMuPDF / pdfplumber → clean_text()
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  PLAN Node    │  Planner Agent → detects headings, infers structure
└──────┬────────┘  Output: [{section_id, title, text}, ...]
       │
       ▼
┌───────────────┐
│  WORK Node    │  N Worker Agents run in parallel (asyncio + Semaphore)
└──────┬────────┘  Output: [{section_id, key_points, summary}, ...]
       │
       ▼
┌───────────────┐
│ REVIEW Node   │  Reviewer Agent → removes redundancy, fixes consistency
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  WRITE Node   │  Final Writer Agent → coherent, well-structured summary
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ OUTPUT Node   │  Attaches timing metadata, returns final result dict
└───────────────┘
```

### Agent Responsibilities

#### 🗺️ Planner Agent (`agents/planner_agent.py`)
- Uses the smart chunker to split the document
- Sends the first ~6000 characters to the LLM to detect headings and infer structure
- Falls back to chunk-based sections if LLM detection fails

#### ⚙️ Worker Agent (`agents/worker_agent.py`)
- Processes **one section at a time** (designed for parallel execution)
- Produces `key_points` (list) + `summary` (text)
- Respects `summary_depth` and `output_mode` from config
- Returns a graceful stub on failure — pipeline never crashes

#### 🔍 Reviewer Agent (`agents/reviewer_agent.py`)
- Receives **all** section summaries as a JSON batch
- Detects redundant key points and merges them
- Standardizes terminology (e.g., "model" vs "algorithm")
- Adds `reviewer_notes` per section and a `global_notes` field

#### ✍️ Final Writer Agent (`agents/final_writer_agent.py`)
- Synthesizes reviewed sections into one coherent summary
- Respects `output_mode`: flowing prose or structured bullet points
- Respects `summary_depth` for target word count

---

## 📤 Output Format

```json
{
  "title": "Introduction to Machine Learning",
  "section_summaries": [
    {
      "section_id": 0,
      "title": "Chapter 1: What is Machine Learning?",
      "summary": "Machine learning is a branch of AI that enables..."
    },
    ...
  ],
  "final_summary": "Machine learning represents one of the most transformative...",
  "_meta": {
    "file": "sample_inputs/sample.txt",
    "num_sections": 5,
    "error": null,
    "timing_seconds": {
      "load": 0.02,
      "plan": 1.45,
      "work": 3.21,
      "review": 1.87,
      "write": 1.62,
      "total": 8.37
    }
  }
}
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short

# Run a specific test class
pytest tests/test_summarizer.py::TestPipelineIntegration -v
```

All tests mock the LLM so **no API key is needed** to run the test suite.

---

## 🔧 Customizing via `config.py`

All system parameters live in `config.py`. You can edit defaults directly or override them via the CLI:

```python
@dataclass
class Config:
    model: str = "gpt-4o-mini"       # swap to "gpt-4o" for higher quality
    temperature: float = 0.2
    max_tokens: int = 4096

    max_chunk_tokens: int = 1500     # reduce for shorter sections
    min_chunk_tokens: int = 100
    overlap_tokens: int = 50

    summary_depth: str = "medium"    # short | medium | detailed
    output_mode: str = "paragraph"   # bullet | paragraph
    max_workers: int = 4             # parallel workers
    log_level: str = "INFO"
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `openai` | LLM API calls |
| `langgraph` | Pipeline state graph orchestration |
| `langchain-core` | Base abstractions for LangGraph |
| `pymupdf` | Fast PDF text extraction |
| `pdfplumber` | Fallback PDF parser for complex layouts |
| `tiktoken` | Token counting for smart chunking |
| `python-dotenv` | Load API key from `.env` |
| `pytest` | Test runner |

---

## 🚫 Design Constraints

- ✅ No RAG — no vector databases, no embeddings, no retrieval
- ✅ No external APIs except OpenAI (or compatible endpoint)
- ✅ No monolithic code — each agent is an independent callable module
- ✅ No hardcoded values — everything flows through `Config`

---

## 🛠️ Troubleshooting

**`OPENAI_API_KEY` not set**
```
AuthenticationError: No API key provided.
```
→ Copy `.env.example` to `.env` and add your key, or `export OPENAI_API_KEY=sk-...`

**PDF text is empty**
→ The PDF may be scanned (image-only). Consider adding an OCR step with `pytesseract`.

**`ModuleNotFoundError: No module named 'fitz'`**
→ Run `pip install pymupdf`

**Rate limit errors**
→ Reduce `--workers` to `2` or `1`, or use a model with higher rate limits.

---

## 📝 License

MIT — free to use, modify, and distribute.
