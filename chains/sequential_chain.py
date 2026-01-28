from tempfile import template

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt1 = PromptTemplate(
    template="generate 100 words report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="extract key points from the following text \n {text}",
    input_variables=["text"],
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "David Lynch"})
print(result)
chain.get_graph().print_ascii()
