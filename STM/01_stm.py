from langgraph.graph import StateGraph, START, MessagesState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


model = ChatOpenAI()


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}



builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")



graph = builder.compile()
graph


graph.invoke({"messages": [{"role": "user", "content": "Hi! My name is Prajwol . I live in Minbhawan ,Kathmandu."}]})


graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]})

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}



builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")



from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)



config = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}


graph.invoke({"messages": [{"role": "user", "content": "Hi! My name is Prajwol ."}]}, config)


graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config2)