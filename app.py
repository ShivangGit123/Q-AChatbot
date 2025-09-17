import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- Page Config ---
st.set_page_config(
    page_title="Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
engine = st.sidebar.selectbox(
    "Select a Model",
    ["llama-3.3-70b-versatile", "openai/gpt-oss-20B", "llama-3.1-8b-instant"]
)
temperature = st.sidebar.slider(" Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider(" Max Tokens", 50, 500, 200)

st.sidebar.markdown("---")
st.sidebar.info("Using **Groq Inference** with LangChain")

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond clearly and concisely."),
    ("user", "Question: {question}")
])

def generate_response(question, engine, temperature, max_tokens):
    llm = ChatGroq(model=engine, temperature=temperature, max_tokens=max_tokens)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"question": question})

# --- Main UI ---
st.title("🤖 Q&A Chatbot")
st.write("Ask me anything and I’ll try to help!")

# Chat input
user_input = st.chat_input("Type your question here...")

# Chat history container
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("Thinking..."):
        response = generate_response(user_input, engine, temperature, max_tokens)

    # Save bot message
    st.session_state["messages"].append({"role": "assistant", "content": response})

# Display conversation
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"👤 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Bot:** {msg['content']}")
