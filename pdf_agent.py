import os
import streamlit as st
from groq import Groq

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(page_title="PDF RAG Agent", page_icon="📄")

# ✅ Load API key - works on Streamlit Cloud and other platforms
groq_api_key = None

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key:
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in secrets or environment variables")
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

# ✅ Cache embeddings only (not LLM)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ✅ Always create fresh LLM with explicit key - NO caching
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0
    )

# ✅ Process PDFs
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


# ✅ Fallback with explicit key
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


# ✅ Upload screen
if not st.session_state.document_uploaded:
    st.title("📄 PDF RAG Agent")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        with st.spinner("Processing PDFs..."):
            process_document(uploaded)
        st.rerun()

# ✅ Chat screen
else:
    st.title("💬 Chat with your PDFs")

    # 🔑 Key test button - remove after confirming it works
    if st.button("🔑 Test Groq Key"):
        try:
            client = Groq(api_key=groq_api_key)
            result = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": "say hi"}],
                max_tokens=10
            )
            st.success(f"✅ Key works! Response: {result.choices[0].message.content}")
        except Exception as e:
            st.error(f"❌ Key test failed: {type(e).__name__}: {str(e)}")

    # Chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask something...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner("Thinking..."):
            answer = None

            # Try agent first
            try:
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": "pdf_chat"}}
                )
                answer = response["messages"][-1].content

            except Exception as agent_error:
                st.warning(f"⚠️ Agent failed: {type(agent_error).__name__}: {str(agent_error)}")

                # Try fallback
                try:
                    answer = rag_fallback(query)

                except Exception as fallback_error:
                    st.error(f"❌ Fallback failed: {type(fallback_error).__name__}: {str(fallback_error)}")
                    answer = "Both agent and fallback failed. Please check the errors above."

        if answer:
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
