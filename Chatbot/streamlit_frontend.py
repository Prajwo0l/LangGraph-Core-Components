import streamlit as st
from langraph_backend import chatbot, retreive_all_threads, delete_thread
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessageChunk, ToolMessage
import uuid

# ------------------------ Utilities ------------------------
def generate_thread_id():
    return str(uuid.uuid4())

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
    """Convert LangChain BaseMessage list into session_state-friendly dicts."""
    result = []
    for message in messages:
        role = 'user' if isinstance(message, HumanMessage) else 'assistant'
        result.append({'role': role, 'content': message.content})
    return result

def handle_delete(thread_id):
    """Delete thread from DB and session state, reset if it was the active chat."""
    delete_thread(thread_id)
    del st.session_state['chat_threads'][thread_id]
    if st.session_state['thread_id'] == thread_id:
        reset_chat()

# ------------------------ Session State Init ------------------------
# Always reload threads from SQLite on every page load / refresh so the
# sidebar reflects what is actually persisted — not a stale in-memory copy.
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retreive_all_threads()

# Restore the active thread.  After a browser refresh session_state is empty,
# so we fall back to the most-recently saved thread from the DB.
if 'thread_id' not in st.session_state:
    threads = st.session_state['chat_threads']
    if threads:
        # Last key = newest thread (list is ordered oldest→newest)
        st.session_state['thread_id'] = list(threads.keys())[-1]
    else:
        st.session_state['thread_id'] = generate_thread_id()

# Make sure the active thread is always registered in the sidebar dict
add_thread(st.session_state['thread_id'])

# Restore message history from SQLite whenever it is missing from session_state.
# This covers the page-refresh case: the DB has the full history even though
# session_state was wiped.
if 'message_history' not in st.session_state:
    messages = load_conversation(st.session_state['thread_id'])
    st.session_state['message_history'] = messages_to_history(messages)

# ------------------------ Display Chat History ------------------------
for message in st.session_state['message_history']:
    if message['content']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# ------------------------ Sidebar ------------------------
st.sidebar.title("SideBar")

if st.sidebar.button('New Chat', key="new_chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id, title in reversed(list(st.session_state['chat_threads'].items())):
    # Skip unsaved new chats that are not currently active
    if title == "New Chat" and thread_id != st.session_state['thread_id']:
        continue

    display_title = title
    if thread_id == st.session_state['thread_id']:
        display_title = '👉' + title

    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
        if st.button(display_title, key=f"thread_{thread_id}"):
            st.session_state['thread_id'] = thread_id
            # Always load from SQLite so switching threads is reliable
            messages = load_conversation(thread_id)
            st.session_state['message_history'] = messages_to_history(messages)
            st.rerun()

    with col2:
        if st.button('🗑️', key=f"delete_{thread_id}"):
            handle_delete(thread_id)
            st.rerun()

# ------------------------ User Input ------------------------
user_input = st.chat_input('Type here')

if user_input:
    current_thread_id = st.session_state['thread_id']
    CONFIG = {'configurable': {'thread_id': current_thread_id},
              'metadata':{
                  'thread_id':current_thread_id},
                  'run_name':'chat_turn',
              }

    # Update title only for brand-new chats
    current_title = st.session_state['chat_threads'][current_thread_id]
    if current_title == "New Chat":
        current_title = user_input.strip()[:30]
        st.session_state['chat_threads'][current_thread_id] = current_title

    # Store user message (no display)
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # Generate and stream assistant response
    response_text = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)], 'title': current_title},
            config=CONFIG,
            stream_mode='messages'
        ):
            # ---- Tool call detected — show status container ----
            if hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                for tool_call in message_chunk.tool_calls:
                    tool_name = tool_call.get('name', 'tool')
                    with status_placeholder.status(f"🔧 Using tool: `{tool_name}`...", expanded=False) as s:
                        st.write(f"**Tool:** `{tool_name}`")
                        st.write(f"**Input:** `{tool_call.get('args', {})}`")
                        s.update(label=f"✅ `{tool_name}` done", state="complete", expanded=False)

            # ---- Tool result — update status with output ----
            elif isinstance(message_chunk, ToolMessage):
                tool_name = metadata.get('name', 'tool')
                with status_placeholder.status(f"✅ `{tool_name}` result received", expanded=False) as s:
                    st.write(f"**Result:** `{message_chunk.content}`")
                    s.update(label=f"✅ `{tool_name}` result received", state="complete", expanded=False)

            # ---- Regular AI text chunk — stream it ----
            else:
                chunk = message_chunk.content
                if chunk:
                    response_text += chunk
                    message_placeholder.markdown(response_text)

    # Store assistant message
    st.session_state['message_history'].append({'role': 'assistant', 'content': response_text})