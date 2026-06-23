import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.title("👧🌸🦋 LangChain Chatbot")
st.caption("Explains everything like you're 2 years old!")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain everything like the user is a 2 year old child. Previous conversation: {history}"),
    ("human", "{input}")
])
chain = prompt | llm | StrOutputParser()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_memory():
    if not st.session_state.chat_history:
        return "No previous conversation."
    return "\n".join([f"{r}: {m}" for r, m in st.session_state.chat_history])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "input": user_input,
                "history": get_memory()
            })
        st.markdown(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))