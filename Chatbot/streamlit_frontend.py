import streamlit as st
from langraph_backend import chatbot, retreive_all_threads, delete_thread, CHATBOT_CONFIG_DEFAULTS
from langchain_core.messages import HumanMessage, ToolMessage
import uuid

# ------------------------ Page Config ------------------------
st.set_page_config(
    page_title="Pattie",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)


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
    """Convert LangChain messages to display-friendly dicts.
    Skips ToolMessages and AI messages that only have tool calls (no text).
    """
    result = []
    for message in messages:
        if isinstance(message, HumanMessage):
            if message.content:
                result.append({'role': 'user', 'content': message.content})
        elif isinstance(message, ToolMessage):
            continue  # skip raw tool output
        else:
            # AIMessage: only keep if it has real text (not just a tool call)
            content = message.content
            if isinstance(content, list):
                text = ' '.join(
                    part.get('text', '') for part in content
                    if isinstance(part, dict) and part.get('type') == 'text'
                )
            else:
                text = content or ''
            if text.strip():
                result.append({'role': 'assistant', 'content': text})
    return result

def handle_delete(thread_id):
    """Delete a thread from DB and session state."""
    delete_thread(thread_id)
    del st.session_state['chat_threads'][thread_id]
    if st.session_state['thread_id'] == thread_id:
        reset_chat()


# ------------------------ Session State Init ------------------------
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retreive_all_threads()

if 'thread_id' not in st.session_state:
    threads = st.session_state['chat_threads']
    st.session_state['thread_id'] = list(threads.keys())[-1] if threads else generate_thread_id()

add_thread(st.session_state['thread_id'])

if 'message_history' not in st.session_state:
    messages = load_conversation(st.session_state['thread_id'])
    st.session_state['message_history'] = messages_to_history(messages)


# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.title('✨ Pattie')
    st.divider()

    if st.button('✏️ New Chat', use_container_width=True, key='new_chat'):
        reset_chat()
        st.rerun()

    st.caption('RECENT CHATS')

    for thread_id, title in reversed(list(st.session_state['chat_threads'].items())):
        if title == 'New Chat' and thread_id != st.session_state['thread_id']:
            continue

        is_active = thread_id == st.session_state['thread_id']
        label = ('**▶ ' if is_active else '') + title + ('**' if is_active else '')

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f'thread_{thread_id}', use_container_width=True):
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)
                st.session_state['message_history'] = messages_to_history(messages)
                st.rerun()
        with col2:
            if st.button('🗑️', key=f'delete_{thread_id}'):
                handle_delete(thread_id)
                st.rerun()


# ------------------------ Main Chat Area ------------------------
st.title('✨ Pattie')
st.caption('Your personal AI assistant')
st.divider()

# Welcome message when chat is empty
if not st.session_state['message_history']:
    st.info('👋 Hello! How can I help you today? Type a message below to get started.')

# Display chat history
for message in st.session_state['message_history']:
    if message['content']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


# ------------------------ User Input ------------------------
user_input = st.chat_input('Ask Pattie...')

if user_input:
    current_thread_id = st.session_state['thread_id']
    CONFIG = {
        'configurable': {'thread_id': current_thread_id},
        'metadata': {'thread_id': current_thread_id},
        'run_name': 'chat_turn',
        **CHATBOT_CONFIG_DEFAULTS,
    }

    # Set thread title from first message
    current_title = st.session_state['chat_threads'][current_thread_id]
    if current_title == 'New Chat':
        current_title = user_input.strip()[:30]
        st.session_state['chat_threads'][current_thread_id] = current_title

    # Show user message immediately
    with st.chat_message('user'):
        st.markdown(user_input)
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # Stream assistant response
    response_text = ''
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)], 'title': current_title},
            config=CONFIG,
            stream_mode='messages'
        ):
            # Tool call — show a status indicator
            if hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                for tool_call in message_chunk.tool_calls:
                    tool_name = tool_call.get('name', 'tool')
                    with status_placeholder.status(f'🔧 Using tool: {tool_name}...', expanded=False) as s:
                        st.write(f'**Tool:** {tool_name}')
                        st.write(f'**Input:** {tool_call.get("args", {})}')
                        s.update(label=f'✅ {tool_name} done', state='complete', expanded=False)

            # Tool result
            elif isinstance(message_chunk, ToolMessage):
                tool_name = metadata.get('name', 'tool')
                with status_placeholder.status(f'✅ {tool_name} result received', expanded=False) as s:
                    st.write(f'**Result:** {message_chunk.content}')
                    s.update(label=f'✅ {tool_name} result received', state='complete', expanded=False)

            # Regular AI text — stream it
            else:
                chunk = message_chunk.content
                if chunk:
                    response_text += chunk
                    message_placeholder.markdown(response_text)

    st.session_state['message_history'].append({'role': 'assistant', 'content': response_text})