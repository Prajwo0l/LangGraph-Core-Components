from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

search = DuckDuckGoSearchRun()
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return str(e)
    
import random
from langchain_core.tools import tool

facts = [
    "Octopus have three hearts",
    "Bananas are berries",
    "Sharks existed before trees",
    "Honey never spoils"
]

@tool
def random_fact() -> str:
    """Returns a random interesting fact."""
    return random.choice(facts)


@tool
def python_docs(question: str) -> str:
    """Answer questions about Python basics."""
    
    docs = {
        "list": "A list is a mutable ordered collection in Python.",
        "tuple": "A tuple is an immutable ordered collection.",
        "dictionary": "A dictionary stores key value pairs."
    }

    for key in docs:
        if key in question.lower():
            return docs[key]

    return "No information found."

tools = [search, calculator, random_fact, python_docs]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

response = agent_executor.invoke(
    {"input": "Who is the director of avatar ?"}
)

print(response["output"])