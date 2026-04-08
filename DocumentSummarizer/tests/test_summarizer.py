"""
tests/test_summarizer.py — Unit & integration tests.

Run with:
    pytest tests/ -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from config import Config
from utils.document_loader import clean_text
from utils.chunker import chunk_document, count_tokens
from agents.planner_agent import run_planner
from agents.worker_agent import run_worker
from agents.reviewer_agent import run_reviewer
from agents.final_writer_agent import run_final_writer


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> Config:
    return Config(
        model="gpt-4o-mini",
        summary_depth="short",
        output_mode="paragraph",
        max_chunk_tokens=500,
        max_workers=2,
    )


SAMPLE_TEXT = """\
Chapter 1: Introduction

Machine learning is a subset of artificial intelligence that enables systems
to learn from data. It has transformed industries ranging from healthcare to
finance.

Chapter 2: Supervised Learning

In supervised learning, models are trained on labelled datasets. Common
algorithms include linear regression, decision trees, and neural networks.

Chapter 3: Unsupervised Learning

Unsupervised learning finds hidden patterns in unlabelled data. Clustering
and dimensionality reduction are key techniques.

Chapter 4: Conclusion

Machine learning continues to evolve rapidly. Future directions include
federated learning and more interpretable models.
"""


# ─── Utils tests ──────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_extra_blank_lines(self):
        text = "Hello\n\n\n\nWorld"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_rejoins_broken_sentences(self):
        text = "The model was trained on a large\ncorpus of text data."
        result = clean_text(text)
        assert "\n" not in result

    def test_strips_control_chars(self):
        text = "Hello\x00World\x07!"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x07" not in result

    def test_empty_string(self):
        assert clean_text("") == ""


class TestChunker:
    def test_produces_chunks(self, cfg):
        chunks = chunk_document(SAMPLE_TEXT, cfg)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "tokens" in chunk
            assert "index" in chunk

    def test_chunk_token_limit(self, cfg):
        chunks = chunk_document(SAMPLE_TEXT, cfg)
        for chunk in chunks:
            assert chunk["tokens"] <= cfg.max_chunk_tokens + cfg.overlap_tokens

    def test_all_text_preserved(self, cfg):
        chunks = chunk_document(SAMPLE_TEXT, cfg)
        combined = " ".join(c["text"] for c in chunks)
        # Check key phrases from the original appear in some chunk
        assert "Machine learning" in combined
        assert "Supervised Learning" in combined

    def test_count_tokens(self):
        tokens = count_tokens("Hello world", "gpt-4o-mini")
        assert tokens > 0


# ─── Agent tests (mocked LLM) ─────────────────────────────────────────────────

MOCK_PLANNER_RESPONSE = {
    "title": "Introduction to Machine Learning",
    "sections": [
        {"section_id": 0, "title": "Chapter 1: Introduction",
         "text": "Machine learning is a subset of artificial intelligence."},
        {"section_id": 1, "title": "Chapter 2: Supervised Learning",
         "text": "In supervised learning, models are trained on labelled datasets."},
    ],
}

MOCK_WORKER_RESPONSE = {
    "section_id": 0,
    "title": "Chapter 1: Introduction",
    "key_points": ["ML is a subset of AI.", "It transforms industries."],
    "summary": "Machine learning enables systems to learn from data.",
}

MOCK_REVIEWER_RESPONSE = {
    "reviewed_sections": [
        {
            "section_id": 0,
            "title": "Chapter 1: Introduction",
            "key_points": ["ML is a subset of AI."],
            "summary": "Machine learning enables systems to learn from data.",
            "reviewer_notes": "No changes needed.",
        }
    ],
    "global_notes": "Document is consistent.",
}

MOCK_WRITER_RESPONSE = {
    "title": "Introduction to Machine Learning",
    "section_summaries": [
        {"section_id": 0, "title": "Introduction",
         "summary": "ML is a subset of AI."},
    ],
    "final_summary": "Machine learning is a transformative field of AI.",
}


class TestPlannerAgent:
    @patch("agents.planner_agent.call_llm", return_value=MOCK_PLANNER_RESPONSE)
    def test_returns_sections(self, mock_llm, cfg):
        result = run_planner(SAMPLE_TEXT, cfg)
        assert "title" in result
        assert "sections" in result
        assert len(result["sections"]) >= 1

    @patch("agents.planner_agent.call_llm", side_effect=RuntimeError("LLM down"))
    def test_fallback_on_llm_failure(self, mock_llm, cfg):
        """Planner should fall back to chunk-based sections on LLM failure."""
        result = run_planner(SAMPLE_TEXT, cfg)
        assert "sections" in result
        assert len(result["sections"]) >= 1


class TestWorkerAgent:
    @patch("agents.worker_agent.call_llm", return_value=MOCK_WORKER_RESPONSE)
    def test_returns_summary(self, mock_llm, cfg):
        section = {"section_id": 0, "title": "Intro", "text": "Some content."}
        result = run_worker(section, cfg)
        assert "summary" in result
        assert "key_points" in result

    def test_empty_section_handled(self, cfg):
        section = {"section_id": 0, "title": "Empty", "text": ""}
        result = run_worker(section, cfg)
        assert result["summary"] == "(Empty section)"

    @patch("agents.worker_agent.call_llm", side_effect=RuntimeError("LLM down"))
    def test_graceful_failure(self, mock_llm, cfg):
        section = {"section_id": 0, "title": "Test", "text": "Content here."}
        result = run_worker(section, cfg)
        assert "failed" in result["summary"].lower()


class TestReviewerAgent:
    @patch("agents.reviewer_agent.call_llm", return_value=MOCK_REVIEWER_RESPONSE)
    def test_returns_reviewed_sections(self, mock_llm, cfg):
        sections = [MOCK_WORKER_RESPONSE]
        result = run_reviewer("Test Doc", sections, cfg)
        assert "reviewed_sections" in result
        assert len(result["reviewed_sections"]) >= 1

    def test_empty_input(self, cfg):
        result = run_reviewer("Test Doc", [], cfg)
        assert result["reviewed_sections"] == []


class TestFinalWriterAgent:
    @patch("agents.final_writer_agent.call_llm", return_value=MOCK_WRITER_RESPONSE)
    def test_returns_final_summary(self, mock_llm, cfg):
        sections = MOCK_REVIEWER_RESPONSE["reviewed_sections"]
        result = run_final_writer("Test Doc", sections, cfg)
        assert "final_summary" in result
        assert len(result["final_summary"]) > 0

    def test_empty_sections(self, cfg):
        result = run_final_writer("Test Doc", [], cfg)
        assert "No content" in result["final_summary"]


# ─── Integration smoke test ───────────────────────────────────────────────────

class TestPipelineIntegration:
    """
    End-to-end smoke test using fully mocked LLM calls.
    Validates that the pipeline wires together correctly.
    """

    @patch("agents.final_writer_agent.call_llm", return_value=MOCK_WRITER_RESPONSE)
    @patch("agents.reviewer_agent.call_llm",    return_value=MOCK_REVIEWER_RESPONSE)
    @patch("agents.worker_agent.call_llm",      return_value=MOCK_WORKER_RESPONSE)
    @patch("agents.planner_agent.call_llm",     return_value=MOCK_PLANNER_RESPONSE)
    def test_full_pipeline_txt(self, m1, m2, m3, m4, tmp_path, cfg):
        from orchestrator.graph import run_pipeline

        # Write a sample TXT file
        doc = tmp_path / "sample.txt"
        doc.write_text(SAMPLE_TEXT, encoding="utf-8")

        result = run_pipeline(str(doc), cfg)
        assert "final_summary" in result
        assert result.get("_meta", {}).get("error") is None

    def test_pipeline_missing_file(self, cfg):
        from orchestrator.graph import run_pipeline
        result = run_pipeline("/nonexistent/path/file.txt", cfg)
        assert result.get("_meta", {}).get("error") is not None
