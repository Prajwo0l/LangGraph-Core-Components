# pattie/memory.py
# =============================================================================
# Short-Term Memory (STM) + Long-Term Memory (LTM) for Pattie.
#
# This is the existing memory.py moved into the pattie/ package.
# The only changes from the original are:
#   • Removed direct OpenAI model instantiation at import time
#     (uses lazy getters instead, so load_dotenv() always runs first).
#   • Path constants now resolve relative to this file's location.
#   • No other logic has changed.
# =============================================================================
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Paths — resolve relative to THIS file so they work regardless of cwd
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).parent
LTM_DIR       = _HERE / "ltm_store"
LTM_FAISS_DIR = LTM_DIR / "faiss_index"
LTM_META_FILE = LTM_DIR / "ltm_meta.json"
STM_DB_FILE   = _HERE / "stm_store.db"

LTM_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy model accessors
# ---------------------------------------------------------------------------
_llm_instance:        Optional[ChatOpenAI]       = None
_embeddings_instance: Optional[OpenAIEmbeddings] = None


def _get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)
    return _llm_instance


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings_instance


# ---------------------------------------------------------------------------
# STM constants
# ---------------------------------------------------------------------------
STM_COMPRESS_AFTER = 10
STM_KEEP_RECENT    = 8

# ---------------------------------------------------------------------------
# STM — per-thread summary store (SQLite)
# ---------------------------------------------------------------------------
_stm_conn = sqlite3.connect(str(STM_DB_FILE), check_same_thread=False)
_stm_conn.execute("""
    CREATE TABLE IF NOT EXISTS stm_summaries (
        thread_id  TEXT PRIMARY KEY,
        summary    TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
""")
_stm_conn.commit()


def _stm_get(thread_id: str) -> Optional[str]:
    row = _stm_conn.execute(
        "SELECT summary FROM stm_summaries WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    return row[0] if row else None


def _stm_set(thread_id: str, summary: str) -> None:
    _stm_conn.execute("""
        INSERT INTO stm_summaries (thread_id, summary, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE
            SET summary    = excluded.summary,
                updated_at = excluded.updated_at
    """, (thread_id, summary, datetime.utcnow().isoformat()))
    _stm_conn.commit()


def _stm_delete(thread_id: str) -> None:
    _stm_conn.execute("DELETE FROM stm_summaries WHERE thread_id = ?", (thread_id,))
    _stm_conn.commit()


def _summarise_messages(messages: List[BaseMessage], existing_summary: Optional[str]) -> str:
    formatted: List[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            formatted.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            formatted.append(f"Assistant: {content}")
    history_text = "\n".join(formatted)
    prev = f"\nExisting summary to extend:\n{existing_summary}\n" if existing_summary else ""
    prompt = (
        "You are a memory compression assistant.\n"
        f"{prev}"
        f"New conversation segment to incorporate:\n{history_text}\n\n"
        "Write a concise running summary (200 words max) capturing:\n"
        "- Main topics discussed\n"
        "- Key decisions or outcomes\n"
        "- Any important context needed to continue the conversation\n"
        "Write in third-person (e.g. 'The user asked about…'). "
        "Do not include greetings or filler."
    )
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def _group_turns(messages: List[BaseMessage]) -> List[List[BaseMessage]]:
    """
    Group messages into atomic turns so we never split an AIMessage from its
    ToolMessages (which would cause OpenAI to reject with a 400 error).
    """
    turns: List[List[BaseMessage]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, (HumanMessage, SystemMessage)):
            turns.append([msg])
            i += 1
        elif isinstance(msg, AIMessage):
            group = [msg]
            i += 1
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                group.append(messages[i])
                i += 1
            turns.append(group)
        elif isinstance(msg, ToolMessage):
            if turns:
                turns[-1].append(msg)
            else:
                turns.append([msg])
            i += 1
        else:
            turns.append([msg])
            i += 1
    return turns


def apply_stm(thread_id: str, messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Return a safe, deduplicated message list to pass to the LLM.

    Steps:
      1. Deduplicate ToolMessages by tool_call_id (keep last occurrence).
      2. Deduplicate AIMessages with identical tool_call id sets.
      3. Group into atomic turns.
      4. Compress oldest turns into a rolling summary when over threshold.
    """
    existing_summary = _stm_get(thread_id)

    # Step 1: deduplicate ToolMessages
    seen_tool_ids: set = set()
    deduped: List[BaseMessage] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tid = getattr(msg, "tool_call_id", None)
            if tid and tid in seen_tool_ids:
                continue
            if tid:
                seen_tool_ids.add(tid)
        deduped.append(msg)
    deduped = list(reversed(deduped))

    # Step 2: deduplicate AIMessages
    seen_ai_call_sets: List[frozenset] = []
    deduped2: List[BaseMessage] = []
    for msg in reversed(deduped):
        if isinstance(msg, AIMessage):
            call_ids = frozenset(c["id"] for c in (getattr(msg, "tool_calls", None) or []))
            if call_ids and call_ids in seen_ai_call_sets:
                continue
            if call_ids:
                seen_ai_call_sets.append(call_ids)
        deduped2.append(msg)
    deduped2 = list(reversed(deduped2))

    # Step 3: group into atomic turns
    turns = _group_turns(deduped2)

    def _is_real(group: List[BaseMessage]) -> bool:
        return any(isinstance(m, (HumanMessage, AIMessage)) for m in group)

    real_count = sum(1 for g in turns if _is_real(g))

    # Step 4: compress if over threshold
    if real_count > STM_COMPRESS_AFTER:
        keep_turns:     List[List[BaseMessage]] = []
        compress_turns: List[List[BaseMessage]] = []
        kept_real = 0
        for group in reversed(turns):
            if kept_real < STM_KEEP_RECENT:
                keep_turns.append(group)
                if _is_real(group):
                    kept_real += 1
            else:
                compress_turns.append(group)
        keep_turns     = list(reversed(keep_turns))
        compress_turns = list(reversed(compress_turns))
        flat_for_summary = [
            m for group in compress_turns for m in group
            if isinstance(m, (HumanMessage, AIMessage))
        ]
        new_summary = _summarise_messages(flat_for_summary, existing_summary)
        _stm_set(thread_id, new_summary)
        flat_keep   = [m for group in keep_turns for m in group]
        summary_msg = SystemMessage(content=f"[Conversation Summary so far]\n{new_summary}")
        return [summary_msg] + flat_keep

    elif existing_summary:
        summary_msg = SystemMessage(content=f"[Conversation Summary so far]\n{existing_summary}")
        return [summary_msg] + deduped2

    else:
        return deduped2


def get_stm_summary(thread_id: str) -> Optional[str]:
    return _stm_get(thread_id)


def clear_stm(thread_id: str) -> None:
    _stm_delete(thread_id)


# ---------------------------------------------------------------------------
# LTM — global cross-thread FAISS + structured profile
# ---------------------------------------------------------------------------

def _load_meta() -> dict:
    if LTM_META_FILE.exists():
        try:
            return json.loads(LTM_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"facts": [], "profile": {}}


def _save_meta(data: dict) -> None:
    LTM_META_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


_ltm_meta: dict = _load_meta()

_faiss_store: Optional[FAISS] = None


def _load_faiss() -> Optional[FAISS]:
    global _faiss_store
    if _faiss_store is not None:
        return _faiss_store
    if LTM_FAISS_DIR.exists() and any(LTM_FAISS_DIR.iterdir()):
        try:
            _faiss_store = FAISS.load_local(
                str(LTM_FAISS_DIR), _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except Exception:
            _faiss_store = None
    return _faiss_store


def _save_faiss(store: FAISS) -> None:
    store.save_local(str(LTM_FAISS_DIR))


def _get_or_create_faiss(initial_docs: Optional[List[Document]] = None) -> FAISS:
    global _faiss_store
    existing = _load_faiss()
    if existing is not None:
        return existing
    placeholder = Document(
        page_content="[memory store initialised]",
        metadata={"type": "system", "fact_id": "__init__"},
    )
    _faiss_store = FAISS.from_documents(initial_docs or [placeholder], _get_embeddings())
    _save_faiss(_faiss_store)
    return _faiss_store


_FACT_EXTRACTION_PROMPT = """\
You are a memory extraction assistant for a personal AI assistant called Pattie.

Read the following conversation exchange and extract ONLY concrete, reusable facts
about the USER (not the assistant). Focus on:
- Personal details (name, location, job, age, family)
- Preferences and habits (likes, dislikes, routines)
- Goals or ongoing projects
- Skills or expertise areas
- Important constraints or requirements they mentioned

Rules:
- Each fact must be ATOMIC (one fact per item, 20 words max).
- Only extract facts that would still be relevant in future conversations.
- Do NOT extract facts about temporary tasks or the current question.
- If there are no useful facts, return an empty list.
- Return ONLY a JSON array of strings. No explanation, no markdown fences.

Conversation:
{exchange}
"""

_PROFILE_UPDATE_PROMPT = """\
You are updating a structured user profile for a personal AI assistant.

Current profile (JSON):
{current_profile}

New facts just extracted:
{new_facts}

Merge the new facts into the profile. Schema:
{{
  "name": "string or null",
  "location": "string or null",
  "occupation": "string or null",
  "preferences": ["list of preference strings"],
  "goals": ["list of ongoing goal strings"],
  "expertise": ["list of skill/expertise strings"],
  "other": ["list of other important facts"]
}}

Rules:
- Only update fields if the new facts provide clear information.
- Append to lists rather than replacing (unless contradicting).
- Return ONLY valid JSON matching the schema above. No explanation.
"""


def _extract_facts(human_msg: str, ai_msg: str) -> List[str]:
    exchange = f"User: {human_msg}\nAssistant: {ai_msg}"
    prompt   = _FACT_EXTRACTION_PROMPT.format(exchange=exchange)
    try:
        response = _get_llm().invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        facts = json.loads(raw)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str) and f.strip()]
    except Exception:
        pass
    return []


def _update_profile(new_facts: List[str]) -> None:
    if not new_facts:
        return
    current = _ltm_meta.get("profile", {})
    prompt  = _PROFILE_UPDATE_PROMPT.format(
        current_profile=json.dumps(current, indent=2),
        new_facts=json.dumps(new_facts),
    )
    try:
        response = _get_llm().invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        updated = json.loads(raw)
        if isinstance(updated, dict):
            _ltm_meta["profile"] = updated
    except Exception:
        pass


def _add_facts_to_faiss(facts: List[str], thread_id: str) -> None:
    docs = [
        Document(
            page_content=fact,
            metadata={
                "type":      "atomic_fact",
                "thread_id": thread_id,
                "added_at":  datetime.utcnow().isoformat(),
                "fact_id":   f"{thread_id}_{i}_{int(datetime.utcnow().timestamp())}",
            },
        )
        for i, fact in enumerate(facts)
    ]
    store = _get_or_create_faiss(docs)
    if docs:
        store.add_documents(docs)
    _save_faiss(store)


# ---------------------------------------------------------------------------
# Public LTM API
# ---------------------------------------------------------------------------

def update_ltm(thread_id: str, human_msg: str, ai_msg: str) -> List[str]:
    facts = _extract_facts(human_msg, ai_msg)
    if not facts:
        return []
    existing  = {f["text"] for f in _ltm_meta.get("facts", [])}
    new_facts = [f for f in facts if f not in existing]
    if not new_facts:
        return []
    for fact in new_facts:
        _ltm_meta.setdefault("facts", []).append({
            "text":      fact,
            "thread_id": thread_id,
            "added_at":  datetime.utcnow().isoformat(),
        })
    _add_facts_to_faiss(new_facts, thread_id)
    _update_profile(new_facts)
    _save_meta(_ltm_meta)
    return new_facts


def retrieve_ltm(query: str, k: int = 5) -> List[str]:
    store = _load_faiss()
    if store is None:
        return [f["text"] for f in _ltm_meta.get("facts", [])][-k:]
    try:
        docs = store.similarity_search(query, k=k)
        return [d.page_content for d in docs if d.metadata.get("fact_id") != "__init__"]
    except Exception:
        return [f["text"] for f in _ltm_meta.get("facts", [])][-k:]


def get_all_ltm_facts() -> List[dict]:
    return list(_ltm_meta.get("facts", []))


def get_ltm_profile() -> dict:
    return dict(_ltm_meta.get("profile", {}))


def delete_ltm_fact(fact_text: str) -> bool:
    facts        = _ltm_meta.get("facts", [])
    original_len = len(facts)
    _ltm_meta["facts"] = [f for f in facts if f["text"] != fact_text]
    if len(_ltm_meta["facts"]) < original_len:
        _save_meta(_ltm_meta)
        _rebuild_faiss()
        return True
    return False


def clear_all_ltm() -> None:
    global _faiss_store
    import shutil
    _ltm_meta["facts"]   = []
    _ltm_meta["profile"] = {}
    _save_meta(_ltm_meta)
    _faiss_store = None
    if LTM_FAISS_DIR.exists():
        shutil.rmtree(LTM_FAISS_DIR)
    LTM_FAISS_DIR.mkdir(exist_ok=True)


def _rebuild_faiss() -> None:
    global _faiss_store
    import shutil
    if LTM_FAISS_DIR.exists():
        shutil.rmtree(LTM_FAISS_DIR)
    LTM_FAISS_DIR.mkdir(exist_ok=True)
    _faiss_store = None
    facts = _ltm_meta.get("facts", [])
    if not facts:
        return
    docs = [
        Document(
            page_content=f["text"],
            metadata={
                "type":      "atomic_fact",
                "thread_id": f.get("thread_id", "unknown"),
                "added_at":  f.get("added_at", ""),
                "fact_id":   f"rebuilt_{i}",
            },
        )
        for i, f in enumerate(facts)
    ]
    _faiss_store = FAISS.from_documents(docs, _get_embeddings())
    _save_faiss(_faiss_store)


def build_memory_context(thread_id: str, messages: List[BaseMessage], query: str) -> str:
    parts: List[str] = []
    facts = retrieve_ltm(query, k=5)
    if facts:
        facts_text = "\n".join(f"- {f}" for f in facts)
        parts.append(f"[Long-Term Memory - relevant facts about the user]\n{facts_text}")
    profile       = get_ltm_profile()
    profile_lines: List[str] = []
    if profile.get("name"):
        profile_lines.append(f"Name: {profile['name']}")
    if profile.get("location"):
        profile_lines.append(f"Location: {profile['location']}")
    if profile.get("occupation"):
        profile_lines.append(f"Occupation: {profile['occupation']}")
    for pref in (profile.get("preferences") or [])[:3]:
        profile_lines.append(f"Preference: {pref}")
    for goal in (profile.get("goals") or [])[:2]:
        profile_lines.append(f"Goal: {goal}")
    if profile_lines:
        parts.append("[User Profile]\n" + "\n".join(profile_lines))
    return "\n\n".join(parts)
