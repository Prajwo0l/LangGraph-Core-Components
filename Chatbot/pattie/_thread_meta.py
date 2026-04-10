# pattie/_thread_meta.py
# =============================================================================
# Tiny helper that other modules can import without creating circular imports.
# rag.py owns the real implementation; this just re-exports the getter.
# =============================================================================
from .rag import get_active_thread

def get_active_thread_metadata() -> str:
    return get_active_thread()
