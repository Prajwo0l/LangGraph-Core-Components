import streamlit as st
from langraph_backend import chatbot, retreive_all_threads, delete_thread, CHATBOT_CONFIG_DEFAULTS, ingest_pdf, set_active_thread, thread_has_document, thread_document_metadata
from langchain_core.messages import HumanMessage, ToolMessage
import uuid

# ------------------------ Page Config ------------------------
st.set_page_config(
    page_title="Pattie",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------ Utilities ------------------------
def generate_thread_id(): return str(uuid.uuid4())

def add_thread(thread_id, title='New Chat'):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = title

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def messages_to_history(messages):
    result = []
    for message in messages:
        if isinstance(message, HumanMessage):
            if message.content:
                result.append({'role': 'user', 'content': message.content})
        elif isinstance(message, ToolMessage):
            continue
        else:
            content = message.content
            if isinstance(content, list):
                text = ' '.join(part.get('text', '') for part in content
                                if isinstance(part, dict) and part.get('type') == 'text')
            else:
                text = content or ''
            if text.strip():
                result.append({'role': 'assistant', 'content': text})
    return result

def handle_delete(thread_id):
    delete_thread(thread_id)
    del st.session_state['chat_threads'][thread_id]
    if st.session_state['thread_id'] == thread_id:
        reset_chat()

# ------------------------ Session State ------------------------
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retreive_all_threads()
if 'thread_id' not in st.session_state:
    threads = st.session_state['chat_threads']
    st.session_state['thread_id'] = list(threads.keys())[-1] if threads else generate_thread_id()
add_thread(st.session_state['thread_id'])
if 'message_history' not in st.session_state:
    msgs = load_conversation(st.session_state['thread_id'])
    st.session_state['message_history'] = messages_to_history(msgs)
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# ------------------------ Theme Colors ------------------------
dark = st.session_state['dark_mode']
bg          = '#0f0f11' if dark else '#f7f7f8'
sidebar_bg  = '#18181b' if dark else '#efefef'
card_bg     = '#1c1c1f' if dark else '#ffffff'
user_bg     = '#2563eb' if dark else '#2563eb'
ai_bg       = '#1e1e22' if dark else '#ffffff'
border      = '#2a2a2e' if dark else '#e2e2e5'
text_main   = '#f4f4f5' if dark else '#18181b'
text_muted  = '#71717a' if dark else '#71717a'
accent      = '#6366f1'

# ------------------------ Global CSS ------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ─ Reset & base ─ */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: {bg} !important;
    font-family: 'DM Sans', sans-serif !important;
    color: {text_main} !important;
}}

/* ─ Hide Streamlit chrome ─ */
#MainMenu, footer, header,
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{ display: none !important; }}

/* ─ Sidebar ─ */
[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    border-right: 1px solid {border} !important;
}}
[data-testid="stSidebar"] * {{ color: {text_main} !important; }}

/* ─ Sidebar buttons ─ */
[data-testid="stSidebar"] .stButton button {{
    background: transparent !important;
    border: none !important;
    color: {text_main} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    width: 100% !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 8px !important;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: {'#27272a' if dark else '#e4e4e7'} !important;
}}

/* ─ Main content area ─ */
.main .block-container {{
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 1.5rem 2rem 7rem !important;
}}

/* ─ Chat messages ─ */
.msg-user {{
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
}}
.msg-user .bubble {{
    background: {user_bg};
    color: #ffffff;
    padding: 0.7rem 1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}}
.msg-ai {{
    display: flex;
    justify-content: flex-start;
    margin: 0.6rem 0;
    gap: 0.6rem;
    align-items: flex-start;
}}
.msg-ai .avatar {{
    width: 30px; height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, {accent}, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 4px;
}}
.msg-ai .bubble {{
    background: {ai_bg};
    color: {text_main};
    padding: 0.7rem 1rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.75;
    border: 1px solid {border};
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}}

/* ─ File chip ─ */
.file-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {'#27272a' if dark else '#f0f0f2'};
    border: 1px solid {border};
    color: {text_muted};
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}}

/* ─ Welcome screen ─ */
.welcome {{
    text-align: center;
    padding: 4rem 1rem 2rem;
}}
.welcome h1 {{
    font-size: 2.4rem;
    font-weight: 600;
    background: linear-gradient(135deg, {accent}, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}}
.welcome p {{
    color: {text_muted};
    font-size: 1rem;
    font-weight: 300;
}}

/* ─ Input form ─ */
[data-testid="stBottom"] {{
    background: {bg} !important;
}}
[data-testid="stForm"] {{
    background: {card_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 16px !important;
    padding: 0.5rem 0.75rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important;
}}
[data-testid="stForm"] textarea {{
    background: transparent !important;
    color: {text_main} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    border: none !important;
    resize: none !important;
    caret-color: {accent} !important;
}}
[data-testid="stForm"] textarea::placeholder {{ color: {text_muted} !important; }}

/* ─ Upload button inside form ─ */
[data-testid="stForm"] [data-testid="stFileUploader"] section {{
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}}
[data-testid="stForm"] [data-testid="stFileUploader"] section button {{
    background: {'#27272a' if dark else '#f0f0f2'} !important;
    border: 1px solid {border} !important;
    border-radius: 8px !important;
    color: {text_muted} !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.6rem !important;
}}
[data-testid="stForm"] [data-testid="stFileUploader"] section small {{
    display: none !important;
}}

/* ─ Send button ─ */
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
    background: {accent} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    transition: opacity 0.15s;
}}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {{
    opacity: 0.85 !important;
}}

/* ─ Status widget ─ */
[data-testid="stStatusWidget"] {{
    background: {card_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    color: {text_muted} !important;
}}

/* ─ Toggle ─ */
[data-testid="stSidebar"] [data-testid="stToggle"] {{
    margin-top: 0.5rem;
}}

/* ─ Scrollbar ─ */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)


# ================================================================
#  SIDEBAR
# ================================================================
thread_id = st.session_state['thread_id']

with st.sidebar:
    st.markdown('<div style="padding: 1.2rem 0.5rem 0.25rem;">', unsafe_allow_html=True)
    st.markdown(f'<span style="font-size:1.3rem; font-weight:600; letter-spacing:-0.02em;">✨ Pattie</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Dark mode toggle
    st.toggle('🌙 Dark mode', key='dark_mode', on_change=st.rerun)

    st.divider()

    if st.button('✟  New Chat', use_container_width=True, key='new_chat'):
        reset_chat()
        st.rerun()

    st.markdown(f'<div style="font-size:0.7rem; color:{text_muted}; text-transform:uppercase; letter-spacing:0.08em; padding: 0.75rem 0.1rem 0.25rem;">Recents</div>', unsafe_allow_html=True)

    for tid, title in reversed(list(st.session_state['chat_threads'].items())):
        if title == 'New Chat' and tid != thread_id:
            continue
        is_active = tid == thread_id
        label = ('▶ ' if is_active else '') + title
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f'thread_{tid}', use_container_width=True):
                st.session_state['thread_id'] = tid
                st.session_state['message_history'] = messages_to_history(load_conversation(tid))
                st.rerun()
        with col2:
            if st.button('🗑️', key=f'delete_{tid}'):
                handle_delete(tid)
                st.rerun()


# ================================================================
#  MAIN AREA
# ================================================================

# ─ File chip if doc loaded ─
if thread_has_document(thread_id):
    meta = thread_document_metadata(thread_id)
    st.markdown(f'<div class="file-chip">📄 {meta.get("filename", "Document")} &nbsp;·&nbsp; {meta.get("documents",0)} pages</div>', unsafe_allow_html=True)

# ─ Welcome screen ─
if not st.session_state['message_history']:
    st.markdown("""
    <div class="welcome">
        <h1>Good to see you</h1>
        <p>Ask me anything — search the web, crunch numbers, query your docs, track expenses.</p>
    </div>
    """, unsafe_allow_html=True)

# ─ Chat history ─
for i, msg in enumerate(st.session_state['message_history']):
    if msg['role'] == 'user':
        st.markdown(f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-ai"><div class="avatar">✨</div><div class="bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)


# ================================================================
#  INPUT FORM  (upload + textarea + send in one row)
# ================================================================
with st.form('chat_form', clear_on_submit=True):
    col_upload, col_text, col_send = st.columns([1, 8, 1])

    with col_upload:
        uploaded_file = st.file_uploader('📎', type='pdf', label_visibility='collapsed')

    with col_text:
        user_input = st.text_area('Message Pattie…', height=68, label_visibility='collapsed',
                                   placeholder='Ask anything… or upload a PDF with 📎')

    with col_send:
        send = st.form_submit_button('➤')

# ─ Handle PDF upload ─
if uploaded_file:
    upload_key = f"{thread_id}:{uploaded_file.name}"
    if st.session_state.get('last_upload_key') != upload_key:
        with st.spinner(f'Processing {uploaded_file.name}…'):
            ingest_pdf(uploaded_file.read(), thread_id, uploaded_file.name)
        st.session_state['last_upload_key'] = upload_key
        st.rerun()

# ─ Handle chat send ─
if send and user_input.strip():
    current_thread_id = st.session_state['thread_id']
    CONFIG = {
        'configurable': {'thread_id': current_thread_id},
        'metadata': {'thread_id': current_thread_id},
        'run_name': 'chat_turn',
        **CHATBOT_CONFIG_DEFAULTS,
    }

    current_title = st.session_state['chat_threads'][current_thread_id]
    if current_title == 'New Chat':
        current_title = user_input.strip()[:30]
        st.session_state['chat_threads'][current_thread_id] = current_title

    set_active_thread(current_thread_id)

    st.session_state['message_history'].append({'role': 'user', 'content': user_input.strip()})

    response_text = ''
    placeholder = st.empty()
    status_placeholder = st.empty()

    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input.strip())], 'title': current_title},
        config=CONFIG,
        stream_mode='messages'
    ):
        # Tool call status
        if hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
            for tool_call in message_chunk.tool_calls:
                tool_name = tool_call.get('name', 'tool')
                with status_placeholder.status(f'🔧 {tool_name}…', expanded=False) as s:
                    st.write(f'**Input:** {tool_call.get("args", {})}')
                    s.update(label=f'✅ {tool_name} done', state='complete', expanded=False)

        # Tool result
        elif isinstance(message_chunk, ToolMessage):
            pass  # handled above

        # AI text stream
        else:
            chunk = message_chunk.content
            if chunk:
                response_text += chunk
                placeholder.markdown(
                    f'<div class="msg-ai"><div class="avatar">✨</div>'
                    f'<div class="bubble">{response_text}▮</div></div>',
                    unsafe_allow_html=True
                )

    # Final render without cursor
    placeholder.markdown(
        f'<div class="msg-ai"><div class="avatar">✨</div>'
        f'<div class="bubble">{response_text}</div></div>',
        unsafe_allow_html=True
    )

    st.session_state['message_history'].append({'role': 'assistant', 'content': response_text})
    st.rerun()