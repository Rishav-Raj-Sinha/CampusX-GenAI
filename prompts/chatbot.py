import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(
            content="answer the questions as if you are an avid redditor who uses r/okbuddyretard on a daily basis"
        )
    ]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        user_input = f"You:{prompt}"
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        result = model.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=result.content))
        st.markdown(result.content)
    st.session_state.messages.append({"role": "assistant", "content": result.content})
