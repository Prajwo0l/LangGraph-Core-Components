import streamlit as st
from langraph_backend import chatbot, retreive_all_threads, delete_thread, CHATBOT_CONFIG_DEFAULTS, ingest_pdf, set_active_thread, thread_has_document, thread_document_metadata
from langchain_core.messages import HumanMessage, ToolMessage
import uuid

# ------------------------ Page Config ------------------------
st.set_page_config(page_title="Pattie", page_icon="✨", layout="wide")

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

# ------------------------ Session ------------------------
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retreive_all_threads()

if 'thread_id' not in st.session_state:
    threads = st.session_state['chat_threads']
    st.session_state['thread_id'] = list(threads.keys())[-1] if threads else generate_thread_id()

add_thread(st.session_state['thread_id'])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.markdown("## ✨ Pattie")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat(); st.rerun()

    st.toggle("🌙 Dark Mode", key="dark_mode")

    st.markdown("---")

    for thread_id, title in reversed(list(st.session_state['chat_threads'].items())):
        col1, col2 = st.columns([5,1])
        with col1:
            if st.button(title, key=f"thread_{thread_id}", use_container_width=True):
                st.session_state['thread_id'] = thread_id
                st.session_state['message_history'] = []
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{thread_id}"):
                delete_thread(thread_id)
                del st.session_state['chat_threads'][thread_id]
                st.rerun()

# ------------------------ Styles ------------------------
bg = "#0E1117" if st.session_state['dark_mode'] else "#FFFFFF"
text = "#FFFFFF" if st.session_state['dark_mode'] else "#000000"
ai_bg = "#1E1E1E" if st.session_state['dark_mode'] else "#F1F0F0"
user_bg = "#2A2A2A" if st.session_state['dark_mode'] else "#DCF8C6"

st.markdown(f"""
<style>
body {{background:{bg}; color:{text};}}
.chat-container {{max-width: 900px; margin:auto;}}
.user {{background:{user_bg}; padding:12px; border-radius:14px; margin:8px 0;}}
.ai {{background:{ai_bg}; padding:12px; border-radius:14px; margin:8px 0;}}
.file-chip {{display:inline-block; padding:6px 10px; background:#444; border-radius:10px; margin:4px; color:white;}}
.copy-btn {{float:right; font-size:12px; cursor:pointer; opacity:0.6;}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# ------------------------ File Chip ------------------------
thread_id = st.session_state['thread_id']

if thread_has_document(thread_id):
    meta = thread_document_metadata(thread_id)
    st.markdown(f'<div class="file-chip">📄 {meta.get("filename")}</div>', unsafe_allow_html=True)

# ------------------------ Chat ------------------------
for i, msg in enumerate(st.session_state['message_history']):
    if msg['role'] == 'user':
        st.markdown(f'<div class="user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai">{msg["content"]}</div>', unsafe_allow_html=True)
        st.button("📋 Copy", key=f"copy_{i}", on_click=lambda x=msg["content"]: st.write(x))

# ------------------------ Input ------------------------
st.markdown("---")

with st.form("chat_form", clear_on_submit=True):
    col0, col1, col2 = st.columns([1,7,1])

    with col0:
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    with col1:
        user_input = st.text_area("Message Pattie...", height=80)

    with col2:
        send = st.form_submit_button("➤")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            ingest_pdf(uploaded_file.read(), thread_id, uploaded_file.name)
        st.rerun()

# ------------------------ Chat Logic ------------------------
if send and user_input.strip():
    CONFIG = {'configurable': {'thread_id': thread_id}, **CHATBOT_CONFIG_DEFAULTS}

    set_active_thread(thread_id)

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    response_text = ""
    placeholder = st.empty()

    for chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode='messages'
    ):
        if isinstance(chunk, ToolMessage): continue
        if chunk.content:
            response_text += chunk.content
            placeholder.markdown(f'<div class="ai">{response_text}</div>', unsafe_allow_html=True)

    st.session_state['message_history'].append({'role': 'assistant', 'content': response_text})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)