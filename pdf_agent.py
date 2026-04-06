import streamlit as st
import os

import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
if not groq_api_key:
    st.error("❌ GROQ_API_KEY missing")
    st.stop()

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key
    )

def process_document(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    if not docs:
        st.error("No PDFs loaded")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    embeddings = load_embeddings()

    vector_db = InMemoryVectorStore.from_documents(
        docs, embeddings
    )

    st.session_state.vector_store = vector_db

    llm = load_llm()

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant document chunks."""
        results = vector_db.similarity_search(query, k=3)

        if not results:
            return "No context found"

        return "\n\n".join([doc.page_content for doc in results])

    system_prompt = """
You are a RAG assistant.

RULES:
1. Always use retrieve_context tool
2. Answer only from retrieved context
3. If not found → say "Not found in document"
"""

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,
        checkpointer=memory,
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

def rag_fallback(query):
    vector_db = st.session_state.vector_store
    llm = load_llm()

    results = vector_db.similarity_search(query, k=3)

    if not results:
        return "No relevant info found"

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Answer using ONLY this context:

{context}

Question: {query}
"""

    return llm.invoke(prompt).content

if not st.session_state.document_uploaded:
    st.title("📄 PDF RAG Agent")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files/"
            os.makedirs(path, exist_ok=True)

            for file in uploaded:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())

            process_document(path)

        st.rerun()

if st.session_state.document_uploaded:
    st.title("💬 Chat with your PDFs")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask something...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner("Thinking..."):
            try:
              
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": "pdf_chat"}}
                )
                answer = response["messages"][-1].content

            except Exception as e:
                answer = rag_fallback(query)

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
