from langgraph.graph import StateGraph ,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import asyncio

import requests
import random


load_dotenv()

llm=ChatOpenAI()



@tool
def calculator(expression: str) -> dict:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression (str): A string containing a basic arithmetic expression 
                          (e.g., "396.73 * 50"). Supports +, -, *, /.

    Returns:
        dict: A dictionary with:
            - 'expression' (str): The expression that was evaluated.
            - 'result' (float): The calculated result.
            - 'error' (str, optional): Error message if the calculation failed.

    """
    try:
        result = float(eval(expression))
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

'''@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. AAPL, TSLA)
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=CE8MY894ND1ESUK5"
    
    r = requests.get(url).json()
    
    try:
        price = float(r["Global Quote"]["05. price"])
        return {"symbol": symbol, "price": price}
    except:
        return {"error": "Could not fetch stock price"}'''
    


#Making tool list
tools=[get_stock_price,search_tool,calculator]

# Make the LLM tool_aware
llm_with_tools=llm.bind_tools(tools)




#state
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def build_graph():
        async def chat_node(state:ChatState):
            '''
            LLM node that may answer or request a tool call.
            '''
            messages=state['messages']
            response=await llm_with_tools.ainvoke(messages)
            return {'messages':[response]}


        tool_node=ToolNode(tools)

        graph=StateGraph(ChatState)
        graph.add_node('chat_node',chat_node)
        graph.add_node('tools',tool_node)

        graph.add_edge(START,'chat_node')
        graph.add_conditional_edges('chat_node',tools_condition)
        graph.add_edge('tools','chat_node')

        chatbot=graph.compile()
        return chatbot

async def main():
    chatbot=build_graph()

    out=await chatbot.ainvoke({'messages':[HumanMessage(content='What is multiplication of 2*4')]})
    print(out['messages'][-1].content)

if __name__=='__main__':
    asyncio.run(main())