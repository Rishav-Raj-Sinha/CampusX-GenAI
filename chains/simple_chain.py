from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt = PromptTemplate(
    template="generate facts about {topic} and format it in a genz brainrot humor",
    input_variables=["topic"],
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Python"})
print(result)
# we can visualize the chain using the following code
chain.get_graph().print_ascii()
