# Import all dependencies
import os
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables: API Key
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Create LLM model for inference from NVIDIA-NIM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None
if "docs_loaded" not in st.session_state:
    st.session_state["docs_loaded"] = False

# Function to create vector embeddings
def vector_embedding():
    st.session_state.embeddings = NVIDIAEmbeddings()
    if not st.session_state["docs_loaded"]:
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Function to truncate documents for token limit
def truncate_documents_for_token_limit(documents, max_tokens=8192):
    total_tokens = 0
    truncated_documents = []
    for doc in documents:
        doc_tokens = len(doc.page_content.split())
        if total_tokens + doc_tokens > max_tokens:
            break
        truncated_documents.append(doc)
        total_tokens += doc_tokens
    return truncated_documents

# Sidebar functionalities
with st.sidebar:
    st.title("RAG App Functionalities")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        st.session_state.loader = PyPDFDirectoryLoader(uploaded_files)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state["docs_loaded"] = True
        st.success("PDF documents uploaded successfully!")

    # Generate embeddings
    if st.button("Generate Document Embeddings"):
        with st.spinner("Embedding documents..."):
            vector_embedding()
        st.success("VectorStoreDB created using NVIDIA embeddings!")

# Main chat interface
st.title("Chat with Your Documents")

# Input for user query
user_query = st.text_input("Enter your question:")

# Process the query when the user presses Enter
if user_query.strip():
    if not st.session_state["vectors"]:
        st.error("Please generate embeddings first!")
    else:
        try:
            # Process the query
            truncated_docs = truncate_documents_for_token_limit(st.session_state.final_documents)
            document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
                Use the context below to answer the question.
                <context>{context}</context>
                Question: {input}
            """))
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Processing your query..."):
                response = retrieval_chain.invoke({'input': user_query, "context": truncated_docs})
            
            # Display bot's response
            st.markdown(f"**Bot:** {response['answer']}")

        except Exception as e:
            st.error(f"Error: {e}")
