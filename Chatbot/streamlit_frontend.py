import streamlit as st
from langraph_backend import chatbot, retreive_all_threads, delete_thread
from langchain_core.messages import HumanMessage
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

def handle_delete(thread_id):
    """Delete thread from DB and session state, reset if it was the active chat."""
    delete_thread(thread_id)
    del st.session_state['chat_threads'][thread_id]
    if st.session_state['thread_id'] == thread_id:
        reset_chat()

# ------------------------ Session State Init ------------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retreive_all_threads()

add_thread(st.session_state['thread_id'])

# ------------------------ Display Chat History ------------------------
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# ------------------------ Sidebar ------------------------
st.sidebar.title("SideBar")

if st.sidebar.button('New Chat', key="new_chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id, title in reversed(list(st.session_state['chat_threads'].items())):
    # Skip empty unsaved chats that are not the active thread
    if title == "New Chat" and thread_id != st.session_state['thread_id']:
        continue
    display_title = title
    if thread_id == st.session_state['thread_id']:
        display_title = '👉' + title

    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
        if st.button(display_title, key=f"thread_{thread_id}"):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            temp_messages = []

            for message in messages:
                role = 'user' if isinstance(message, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': message.content})

            st.session_state['message_history'] = temp_messages
            st.rerun()

    with col2:
        if st.button('🗑️', key=f"delete_{thread_id}"):
            handle_delete(thread_id)
            st.rerun()

# ------------------------ User Input ------------------------
user_input = st.chat_input('Type here')

if user_input:
    current_thread_id = st.session_state['thread_id']
    CONFIG = {'configurable': {'thread_id': current_thread_id}}

    # Update title only for new chat
    current_title = st.session_state['chat_threads'][current_thread_id]
    if current_title == "New Chat":
        current_title = user_input.strip()[:30]
        st.session_state['chat_threads'][current_thread_id] = current_title

    # Store user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Generate assistant response
    response_text = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)], 'title': current_title},
            config=CONFIG,
            stream_mode='messages'
        ):
            chunk = message_chunk.content
            if chunk:
                response_text += chunk
                message_placeholder.markdown(response_text)

    # Store assistant message
    st.session_state['message_history'].append({'role': 'assistant', 'content': response_text})