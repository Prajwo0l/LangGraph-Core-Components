# =============================================================================
# streamlit_frontend.py  —  Pattie AI Assistant Frontend
# =============================================================================
import csv
import io
import json
import sqlite3
import uuid
from datetime import date

import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage

from langraph_backend import (
    CHATBOT_CONFIG_DEFAULTS,
    chatbot,
    delete_thread,
    execute_tool_call,
    ingest_pdf,
    retreive_all_threads,
    set_active_thread,
    thread_document_metadata,
    thread_has_document,
    TOOL_GROUPS,
    tools,
    _invoke_tool,
)

# Path to the expense database
EXPENSE_DB = r'C:\Users\lamic\Desktop\Expense MCP Server\expenses.db'


# =============================================================================
# Expense helpers
# =============================================================================
def _get_monthly_summary(year: int, month: int) -> list[dict]:
    start = f'{year}-{month:02d}-01'
    end   = f'{year + 1}-01-01' if month == 12 else f'{year}-{month + 1:02d}-01'
    try:
        with sqlite3.connect(EXPENSE_DB) as conn:
            rows = conn.execute(
                'SELECT category, SUM(amount) AS total FROM expenses '
                'WHERE date >= ? AND date < ? GROUP BY category ORDER BY total DESC',
                (start, end),
            ).fetchall()
        return [{'category': r[0], 'total': r[1]} for r in rows]
    except Exception:
        return []


def _get_all_expenses(year: int, month: int) -> list[dict]:
    start = f'{year}-{month:02d}-01'
    end   = f'{year + 1}-01-01' if month == 12 else f'{year}-{month + 1:02d}-01'
    try:
        with sqlite3.connect(EXPENSE_DB) as conn:
            cur = conn.execute(
                'SELECT id, date, amount, category, subcategory, note FROM expenses '
                'WHERE date >= ? AND date < ? ORDER BY date ASC',
                (start, end),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        return []


def _to_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b''
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode('utf-8')


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title='Pattie',
    page_icon='✨',
    layout='centered',
    initial_sidebar_state='expanded',
)


# =============================================================================
# Session state helpers
# =============================================================================
def _new_thread_id() -> str:
    return str(uuid.uuid4())


def _register_thread(tid: str, title: str = 'New Chat') -> None:
    if tid not in st.session_state.chat_threads:
        st.session_state.chat_threads[tid] = title


def _reset_chat() -> None:
    tid = _new_thread_id()
    st.session_state.thread_id = tid
    _register_thread(tid)
    st.session_state.message_history = []
    st.session_state.pop('last_upload_key', None)
    # Clear any pending HITL
    st.session_state.hitl_pending    = None
    st.session_state.fs_hitl_pending = None


def _load_history(tid: str) -> list:
    state  = chatbot.get_state(config={'configurable': {'thread_id': tid}})
    result = []
    for msg in state.values.get('messages', []):
        if isinstance(msg, HumanMessage) and msg.content:
            result.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, ToolMessage):
            continue
        else:
            content = msg.content
            if isinstance(content, list):
                text = ' '.join(
                    p.get('text', '') for p in content
                    if isinstance(p, dict) and p.get('type') == 'text'
                )
            else:
                text = content or ''
            if text.strip():
                result.append({'role': 'assistant', 'content': text})
    return result


def _delete_thread(tid: str) -> None:
    delete_thread(tid)
    st.session_state.chat_threads.pop(tid, None)
    if st.session_state.thread_id == tid:
        _reset_chat()


def _get_config(tid: str, title: str) -> dict:
    return {
        'configurable': {'thread_id': tid},
        'metadata':     {'thread_id': tid},
        'run_name':     'chat_turn',
        **CHATBOT_CONFIG_DEFAULTS,
    }


# =============================================================================
# Session state bootstrap
# =============================================================================
if 'hitl_pending'    not in st.session_state:
    st.session_state.hitl_pending    = None   # expense → calendar approval
if 'fs_hitl_pending' not in st.session_state:
    st.session_state.fs_hitl_pending = None   # filesystem write/delete approval

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = retreive_all_threads()

if 'thread_id' not in st.session_state:
    threads = st.session_state.chat_threads
    st.session_state.thread_id = list(threads.keys())[-1] if threads else _new_thread_id()

_register_thread(st.session_state.thread_id)

if 'message_history' not in st.session_state:
    st.session_state.message_history = _load_history(st.session_state.thread_id)


# =============================================================================
# Sidebar
# =============================================================================
tid = st.session_state.thread_id

with st.sidebar:
    st.title('✨ Pattie')
    st.divider()

    if st.button('✏️ New Chat', use_container_width=True, key='new_chat'):
        _reset_chat()
        st.rerun()

    st.caption('RECENT CHATS')

    for _tid, _title in reversed(list(st.session_state.chat_threads.items())):
        if _title == 'New Chat' and _tid != tid:
            continue
        label = ('▶ ' if _tid == tid else '') + _title
        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(label, key=f'thread_{_tid}', use_container_width=True):
                st.session_state.thread_id      = _tid
                st.session_state.message_history = _load_history(_tid)
                st.session_state.fs_hitl_pending = None
                st.session_state.hitl_pending    = None
                st.rerun()
        with c2:
            if st.button('🗑️', key=f'del_{_tid}'):
                _delete_thread(_tid)
                st.rerun()


# =============================================================================
# Main area
# =============================================================================
st.title('✨ Pattie')
st.caption('Your personal AI assistant — chat, search, track expenses, read documents.')
st.divider()

tab_chat, tab_dashboard = st.tabs(['💬 Chat', '📊 Expense Dashboard'])


# =============================================================================
# TAB 1 — Chat
# =============================================================================
with tab_chat:

    # ── PDF uploader ──────────────────────────────────────────────────────────
    with st.expander('📎 Upload a PDF', expanded=not thread_has_document(tid)):
        if thread_has_document(tid):
            meta = thread_document_metadata(tid)
            st.success(
                f"📄 **{meta.get('filename')}** — "
                f"{meta.get('documents', 0)} pages, {meta.get('chunks', 0)} chunks ready."
            )
        pdf_file = st.file_uploader('Choose a PDF', type='pdf', label_visibility='collapsed')
        if pdf_file:
            key = f'{tid}:{pdf_file.name}'
            if st.session_state.get('last_upload_key') != key:
                with st.spinner(f'Processing {pdf_file.name}…'):
                    try:
                        ingest_pdf(pdf_file.read(), tid, pdf_file.name)
                        st.session_state.last_upload_key = key
                    except Exception as exc:
                        st.error(f'Failed to process PDF: {exc}')
                st.rerun()

    # ── Welcome message ───────────────────────────────────────────────────────
    if not st.session_state.message_history:
        st.info(
            '👋 Hello! I can search the web, do maths, look up stock prices, '
            'track your expenses (+ Google Calendar), manage files in your Downloads folder, '
            'or answer questions from a PDF. Just ask!'
        )

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.message_history:
        if msg.get('content'):
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    # =========================================================================
    # HITL 1 — Expense → Google Calendar approval
    # =========================================================================
    if st.session_state.hitl_pending:
        expense = st.session_state.hitl_pending
        with st.container(border=True):
            st.markdown('### 📅 Add to Google Calendar?')
            st.write(
                f"Pattie wants to add **{expense['category']}** — "
                f"**Rs. {expense['amount']}** on **{expense['date']}** to your calendar."
            )
            if expense.get('note'):
                st.caption(f"Note: {expense['note']}")

            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button('✅ Yes, add to calendar', use_container_width=True, type='primary'):
                    with st.spinner('Adding to Google Calendar…'):
                        cal_tool = next((t for t in tools if t.name == 'add_to_calendar'), None)
                        if cal_tool is None:
                            st.error('⚠️ Calendar tool not found.')
                        else:
                            try:
                                import asyncio as _asyncio
                                _asyncio.run(_invoke_tool(cal_tool, {
                                    'date':     expense['date'],
                                    'amount':   expense['amount'],
                                    'category': expense['category'],
                                    'note':     expense.get('note', ''),
                                }))
                                st.success('📅 Calendar event created!')
                            except Exception as exc:
                                st.error(f'Calendar error: {exc}')
                    st.session_state.hitl_pending = None
                    st.rerun()

            with col_reject:
                if st.button('❌ No, skip calendar', use_container_width=True):
                    st.session_state.hitl_pending = None
                    st.info('Expense saved. Calendar skipped.')
                    st.rerun()

    # =========================================================================
    # HITL 2 — Filesystem write_file / delete_file approval
    # =========================================================================
    if st.session_state.fs_hitl_pending:
        pending      = st.session_state.fs_hitl_pending
        tool_name    = pending['tool_name']
        args         = pending['args']
        tool_call_id = pending['tool_call_id']
        current_tid  = st.session_state.thread_id

        with st.container(border=True):
            if tool_name == 'write_file':
                st.markdown('### 🗂️ Confirm File Write')
                st.write(f"Pattie wants to write to **`{args.get('path')}`** in your Downloads folder.")
                content_preview = args.get('content', '')
                if len(content_preview) > 300:
                    content_preview = content_preview[:300] + '\n…(truncated)'
                st.code(content_preview, language='text')
                approve_label = '✅ Yes, write the file'
                reject_label  = '❌ No, cancel'
                warning_note  = '⚠️ This will create or overwrite the file.'

            elif tool_name == 'delete_file':
                st.markdown('### 🗑️ Confirm File Deletion')
                st.write(f"Pattie wants to **permanently delete** the file `{args.get('path')}` from your Downloads folder.")
                approve_label = '✅ Yes, delete it'
                reject_label  = '❌ No, keep it'
                warning_note  = '🚨 This cannot be undone.'

            else:  # delete_folder
                st.markdown('### 🗁️ Confirm Folder Deletion')
                st.write(f"Pattie wants to **permanently delete the folder** `{args.get('path')}` and **all its contents** from your Downloads folder.")
                approve_label = '✅ Yes, delete the folder'
                reject_label  = '❌ No, keep it'
                warning_note  = '🚨 This will delete the folder AND everything inside it. This cannot be undone.'

            st.caption(warning_note)
            col_a, col_r = st.columns(2)

            with col_a:
                if st.button(approve_label, use_container_width=True, type='primary'):
                    with st.spinner(f'Running {tool_name}…'):
                        # 1. Execute the tool directly
                        result_str = execute_tool_call(tool_name, args)

                        # 2. Inject the real result back into LangGraph state
                        #    so the graph knows the tool completed and can continue
                        config = _get_config(current_tid, '')
                        chatbot.update_state(
                            config,
                            {
                                'messages':       [ToolMessage(content=result_str, tool_call_id=tool_call_id)],
                                'fs_hitl_pending': None,
                            },
                            as_node='fs_hitl',
                        )

                        # 3. Resume streaming so the LLM produces a final response
                        resume_text = ''
                        with st.chat_message('assistant'):
                            placeholder = st.empty()
                            for chunk, _ in chatbot.stream(
                                None,              # None = resume from current state
                                config=config,
                                stream_mode='messages',
                            ):
                                if hasattr(chunk, 'content') and chunk.content and not isinstance(chunk, ToolMessage):
                                    resume_text += chunk.content
                                    placeholder.markdown(resume_text)

                        if resume_text:
                            st.session_state.message_history.append(
                                {'role': 'assistant', 'content': resume_text}
                            )

                    st.session_state.fs_hitl_pending = None
                    st.rerun()

            with col_r:
                if st.button(reject_label, use_container_width=True):
                    # Inject a cancelled ToolMessage so LangGraph state is consistent
                    config = _get_config(current_tid, '')
                    cancel_msg = f"Action cancelled by user. Do not attempt {tool_name} again unless explicitly asked."
                    chatbot.update_state(
                        config,
                        {
                            'messages':       [ToolMessage(content=cancel_msg, tool_call_id=tool_call_id)],
                            'fs_hitl_pending': None,
                        },
                        as_node='fs_hitl',
                    )

                    # Resume so the LLM acknowledges the cancellation
                    resume_text = ''
                    with st.chat_message('assistant'):
                        placeholder = st.empty()
                        for chunk, _ in chatbot.stream(
                            None,
                            config=config,
                            stream_mode='messages',
                        ):
                            if hasattr(chunk, 'content') and chunk.content and not isinstance(chunk, ToolMessage):
                                resume_text += chunk.content
                                placeholder.markdown(resume_text)

                    if resume_text:
                        st.session_state.message_history.append(
                            {'role': 'assistant', 'content': resume_text}
                        )

                    st.session_state.fs_hitl_pending = None
                    st.rerun()

    # =========================================================================
    # Chat input
    # =========================================================================
    user_input = st.chat_input('Ask Pattie…')

    if user_input:
        current_tid   = st.session_state.thread_id
        current_title = st.session_state.chat_threads[current_tid]
        if current_title == 'New Chat':
            current_title = user_input.strip()[:30]
            st.session_state.chat_threads[current_tid] = current_title

        config = _get_config(current_tid, current_title)

        set_active_thread(current_tid)

        # Clear any stale HITL state before new turn
        st.session_state.hitl_pending    = None
        st.session_state.fs_hitl_pending = None
        new_expense_hitl = None
        new_fs_hitl      = None

        with st.chat_message('user'):
            st.markdown(user_input)
        st.session_state.message_history.append({'role': 'user', 'content': user_input})

        response_text = ''
        with st.chat_message('assistant'):
            msg_placeholder    = st.empty()
            status_placeholder = st.empty()
            intent_placeholder = st.empty()

            intent_icons = {
                'expense':    '💸',
                'filesystem': '🗂️',
                'search':     '🔍',
                'document':   '📄',
                'finance':    '📈',
                'general':    '💬',
                'pdf_ingest': '📥',
            }

            try:
                for chunk, _meta in chatbot.stream(
                    {
                        'messages':       [HumanMessage(content=user_input)],
                        'title':          current_title,
                        'intent':         'general',
                        'fs_hitl_pending': None,
                    },
                    config=config,
                    stream_mode='messages',
                ):
                    # Intent badge
                    if hasattr(chunk, 'intent') and chunk.intent:
                        icon       = intent_icons.get(chunk.intent, '🧠')
                        tool_count = len(TOOL_GROUPS.get(chunk.intent, []))
                        intent_placeholder.caption(
                            f'{icon} **{chunk.intent}** mode — {tool_count} tool(s) available'
                        )

                    # Tool call status
                    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            name = tc.get('name', 'tool')
                            # Show a different label for HITL tools
                            if name in ('write_file', 'delete_file'):
                                label = f'⏸️ {name} — waiting for approval…'
                            else:
                                label = f'🔧 {name}…'
                            with status_placeholder.status(label, expanded=False) as s:
                                st.write(f'**Args:** {tc.get("args", {})}')
                                s.update(label=f'✅ {name} staged', state='complete', expanded=False)

                    elif isinstance(chunk, ToolMessage):
                        # ── Detect expense HITL (calendar) ───────────────────
                        try:
                            result = json.loads(chunk.content)
                            if (isinstance(result, dict)
                                    and result.get('status') == 'ok'
                                    and 'category' in result):
                                new_expense_hitl = {
                                    'date':     result.get('date', ''),
                                    'amount':   result.get('amount', 0),
                                    'category': result.get('category', ''),
                                    'note':     result.get('note', ''),
                                }
                        except Exception:
                            pass

                    elif chunk.content:
                        response_text += chunk.content
                        msg_placeholder.markdown(response_text)

            except Exception as exc:
                error_msg = f'⚠️ Something went wrong: {exc}'
                msg_placeholder.error(error_msg)
                response_text = error_msg

        st.session_state.message_history.append({'role': 'assistant', 'content': response_text})

        # ── After stream: check LangGraph state for fs_hitl_pending ──────────
        graph_state = chatbot.get_state(config)
        fs_pending  = graph_state.values.get('fs_hitl_pending')
        if fs_pending:
            st.session_state.fs_hitl_pending = fs_pending
            st.rerun()

        # ── After stream: set expense calendar HITL if detected ──────────────
        if new_expense_hitl:
            st.session_state.hitl_pending = new_expense_hitl
            st.rerun()


# =============================================================================
# TAB 2 — Expense Dashboard
# =============================================================================
with tab_dashboard:
    st.subheader('📊 Monthly Expense Summary')

    today  = date.today()
    MONTHS = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    col_y, col_m = st.columns(2)
    with col_y:
        selected_year = st.selectbox(
            'Year', options=list(range(today.year, today.year - 5, -1)), index=0
        )
    with col_m:
        selected_month = st.selectbox(
            'Month', options=list(range(1, 13)),
            format_func=lambda m: MONTHS[m - 1],
            index=today.month - 1,
        )

    summary  = _get_monthly_summary(selected_year, selected_month)
    all_rows = _get_all_expenses(selected_year, selected_month)

    if not summary:
        st.info(f'No expenses found for {MONTHS[selected_month - 1]} {selected_year}.')
    else:
        total = sum(r['total'] for r in summary)
        st.metric(
            label=f'Total spent in {MONTHS[selected_month - 1]} {selected_year}',
            value=f'Rs. {total:,.2f}',
        )
        st.divider()

        st.caption('Spending by category')
        st.bar_chart({r['category']: r['total'] for r in summary}, use_container_width=True)
        st.divider()

        st.caption('Category breakdown')
        for r in summary:
            pct = (r['total'] / total * 100) if total else 0
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.write(f"**{r['category']}**")
            c2.write(f"Rs. {r['total']:,.2f}")
            c3.write(f'{pct:.1f}%')

        st.divider()
        st.caption('Export')
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                '⬇️ Download Summary CSV',
                data=_to_csv_bytes(summary),
                file_name=f'expenses_summary_{selected_year}_{selected_month:02d}.csv',
                mime='text/csv',
                use_container_width=True,
            )
        with col_exp2:
            st.download_button(
                '⬇️ Download Full Detail CSV',
                data=_to_csv_bytes(all_rows),
                file_name=f'expenses_detail_{selected_year}_{selected_month:02d}.csv',
                mime='text/csv',
                use_container_width=True,
            )
