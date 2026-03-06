from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


os.environ['LANGCHAIN_PROJECT']='Sequential LLM'



load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variable=['topic']
)

prompt2=PromptTemplate(
    template='Generate a 5 pointer summary from the follwoing text \n {text}',
    input_variables=['text']
)

model=ChatOpenAI(model='gpt-4o-mini',temperature=0.7)
model2=ChatOpenAI(model='gpt-4o',temperature=0.5)

parser=StrOutputParser()

chain=prompt1|model|parser|prompt2|model2|parser

config={
    'run_name':'sequential_chain',
    'tags':['llm app','report generation','summarization'],
    'metadata':{'model':'gpt-40-mini','model2':'gpt-4o'}
}

result=chain.invoke({'topic':"AI Bubble across the Word"},config=config)
print(result)