from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st

st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("Chatbot Básico con LangChain")
st.markdown("Este es un Chatbot de Ejemplo construido con LangChain y Streamlit, por Ernie Calderon. Escribe tu mensaje para comenzar")

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)