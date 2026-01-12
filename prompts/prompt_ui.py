import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.header("research tool")

# userinput = st.text_input("Enter your query here")
paper_input = st.text_input("Enter the paper title")
style_input = st.selectbox(
    "Select the explanation style", ["Simple", "Detailed", "Technical"]
)
length_input = st.selectbox(
    "Select the explanation length", ["Short", "Medium", "Long"]
)

template = load_prompt("template.json")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

if st.button("search"):
    st.write("Searching...")
    # creating a basic chain
    # the input if given for first component of the chain
    # then the ouput of the said component is given to the second component as input automatically by langchain
    chain = template | model
    result = chain.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input,
        }
    )
    st.write(result.content)
