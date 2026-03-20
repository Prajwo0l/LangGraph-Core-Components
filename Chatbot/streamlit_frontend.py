import streamlit as st
from langraph_backend import chatbot, retreive_all_threads, delete_thread, CHATBOT_CONFIG_DEFAULTS, ingest_pdf, set_active_thread, thread_has_document, thread_document_metadata
from langchain_core.messages import HumanMessage, ToolMessage
import uuid

st.set_page_config(
    page_title="Pattie",
    page_icon="✨",
    layout="centered",
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

# ================================================================
#  SIDEBAR
# ================================================================
thread_id = st.session_state['thread_id']

with st.sidebar:
    st.title('✨ Pattie')
    st.divider()

    if st.button('✏️ New Chat', use_container_width=True, key='new_chat'):
        reset_chat()
        st.rerun()

    st.caption('RECENT CHATS')

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
st.title('✨ Pattie')
st.caption('Your personal AI assistant — chat, search, track expenses, read documents.')
st.divider()

# ─ PDF upload section ─
with st.expander('📎 Upload a PDF', expanded=not thread_has_document(thread_id)):
    if thread_has_document(thread_id):
        meta = thread_document_metadata(thread_id)
        st.success(f"📄 **{meta.get('filename')}** — {meta.get('documents', 0)} pages, {meta.get('chunks', 0)} chunks loaded")
    uploaded_file = st.file_uploader('Choose a PDF', type='pdf', label_visibility='collapsed')
    if uploaded_file:
        upload_key = f"{thread_id}:{uploaded_file.name}"
        if st.session_state.get('last_upload_key') != upload_key:
            with st.spinner(f'Processing {uploaded_file.name}…'):
                ingest_pdf(uploaded_file.read(), thread_id, uploaded_file.name)
            st.session_state['last_upload_key'] = upload_key
            st.rerun()

# ─ Welcome message ─
if not st.session_state['message_history']:
    st.info('👋 Hello! Ask me anything — I can search the web, do math, track your expenses, or answer questions from an uploaded PDF.')

# ─ Chat history ─
for msg in st.session_state['message_history']:
    if msg['content']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

# ================================================================
#  USER INPUT
# ================================================================
user_input = st.chat_input('Ask Pattie...')

if user_input:
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

    with st.chat_message('user'):
        st.markdown(user_input)
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    response_text = ''
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)], 'title': current_title},
            config=CONFIG,
            stream_mode='messages'
        ):
            if hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                for tool_call in message_chunk.tool_calls:
                    tool_name = tool_call.get('name', 'tool')
                    with status_placeholder.status(f'🔧 Using {tool_name}...', expanded=False) as s:
                        st.write(f'**Input:** {tool_call.get("args", {})}')
                        s.update(label=f'✅ {tool_name} done', state='complete', expanded=False)
            elif isinstance(message_chunk, ToolMessage):
                pass
            else:
                chunk = message_chunk.content
                if chunk:
                    response_text += chunk
                    message_placeholder.markdown(response_text)

    st.session_state['message_history'].append({'role': 'assistant', 'content': response_text})