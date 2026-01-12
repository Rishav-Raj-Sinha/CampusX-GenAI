import streamlit as st

st.header("research tool")

userinput = st.text_input("Enter your query here")

if st.button("search"):
    st.write("Searching...")
