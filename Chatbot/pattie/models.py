# pattie/models.py
# =============================================================================
# Instantiates the LLM(s) used by the chatbot.
# Keeping model construction here means the rest of the code stays clean and
# you only have one place to swap out models or change parameters.
# =============================================================================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Primary assistant model
llm = ChatOpenAI(model="gpt-4o-mini", max_retries=2)

# Low-temperature model for deterministic classification tasks (intent router)
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=2)

# Embedding model used by RAG and LTM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
