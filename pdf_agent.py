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

st.set_page_config(page_title="PDF RAG Agent", page_icon="📄")

# ✅ Load API key
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY missing in Streamlit secrets")
    st.stop()

# ✅ Session state
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

# ✅ Key change: pass api_key explicitly, underscore prefix skips hashing
@st.cache_resource
def load_llm(_api_key: str):
    return ChatGroq(
        model="llama3-70b-8192",
        api_key=_api_key
    )

def process_document(uploaded_files):
    os.makedirs("temp_docs", exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join("temp_docs", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

    loader = PyPDFDirectoryLoader("temp_docs")
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
    vector_db = InMemoryVectorStore.from_documents(docs, embeddings)
    st.session_state.vector_store = vector_db

    # ✅ Pass key explicitly every time
    llm = load_llm(groq_api_key)

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
    # ✅ Pass key explicitly
    llm = load_llm(groq_api_key)

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


# ✅ UI
if not st.session_state.document_uploaded:
    st.title("📄 PDF RAG Agent")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        with st.spinner("Processing..."):
            process_document(uploaded)
        st.rerun()

else:
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
            except Exception:
                answer = rag_fallback(query)

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
