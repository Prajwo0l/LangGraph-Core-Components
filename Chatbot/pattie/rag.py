# pattie/rag.py
# =============================================================================
# PDF ingestion and per-thread FAISS retriever management.
# The chat node calls rag_tool (defined in tools/builtin.py) which looks up
# the retriever stored here by thread_id.
# =============================================================================
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from .models import embeddings

# ── In-memory retriever store (keyed by thread_id) ───────────────────────────
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA:   Dict[str, dict] = {}
_ACTIVE_THREAD_ID:  Optional[str] = None


def set_active_thread(thread_id: str) -> None:
    global _ACTIVE_THREAD_ID
    _ACTIVE_THREAD_ID = str(thread_id)


def get_active_thread() -> str:
    return _ACTIVE_THREAD_ID or "default"


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def get_retriever(thread_id: str) -> Optional[Any]:
    return _THREAD_RETRIEVERS.get(str(thread_id))


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Chunk a PDF, embed it, and store the retriever keyed by thread_id.
    Returns metadata dict: {filename, documents, chunks}.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        ).split_documents(docs)
        retriever = FAISS.from_documents(chunks, embeddings).as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(tmp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
