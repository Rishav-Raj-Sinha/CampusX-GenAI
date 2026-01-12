import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

# template
template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1. Mathematical Details:
       - Include relevant mathematical equations if present in the paper.
       - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies:
       - Use relatable analogies to simplify complex ideas.
    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
    Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)

prompt = template.invoke(
    {
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
    }
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

if st.button("search"):
    st.write("Searching...")
    result = model.invoke(prompt)
    st.write(result.content)
