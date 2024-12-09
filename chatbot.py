# Import all dependencies
import os
import time
import json
import numpy as np
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
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
if "history" not in st.session_state:
    st.session_state["history"] = []
if "loader" not in st.session_state:
    st.session_state["loader"] = PyPDFDirectoryLoader("./data")
if "docs_loaded" not in st.session_state:
    st.session_state["docs_loaded"] = False

# Function to create vector embeddings
def vector_embedding():
    st.session_state.embeddings = NVIDIAEmbeddings()
    if not st.session_state["docs_loaded"]:
        st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Function to visualize embeddings with metadata
def visualize_embeddings(vectors):
    embeddings = np.array([vectors.index.reconstruct(i) for i in range(vectors.index.ntotal)])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Retrieve metadata for chunks
    metadata = [doc.metadata for doc in st.session_state.final_documents]

    plt.figure(figsize=(10, 6))
    for idx, coord in enumerate(reduced_embeddings):
        plt.scatter(coord[0], coord[1], alpha=0.7, label=f"Chunk {idx}: {metadata[idx].get('source', 'Unknown')}")

    plt.title("t-SNE Visualization of Document Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(plt)

# Function to summarize documents
def summarize_documents(summary_type="high_level"):
    from langchain.chains.summarize import load_summarize_chain
    summary_chain = load_summarize_chain(llm, chain_type="stuff")

    summaries = []
    try:
        for doc in st.session_state.final_documents:
            # Create a prompt based on the summary type
            prompt_text = f"Summarize this in detail: {doc.page_content}" if summary_type == "detailed" else f"Provide a high-level overview: {doc.page_content}"
            
            # Invoke the summary chain
            chunk_summary = summary_chain.invoke({"input_documents": [doc]})
            
            # Extract the text content (assuming the response is a dictionary with a 'text' key)
            summaries.append(chunk_summary.get("text", "No summary available"))
        
        # Join the summaries into a single string
        return "\n".join(summaries)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Failed to summarize documents."


# Function to check token limits
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

    # Summarization
    summary_type = st.selectbox("Summarization Type", options=["High-Level", "Detailed"])
    if st.button("Summarize Documents"):
        if not st.session_state["vectors"]:
            st.error("Please generate embeddings first!")
        else:
            with st.spinner("Summarizing documents..."):
                summary = summarize_documents(summary_type.lower())
            st.write("Summary of Documents:")
            st.write(summary)

    # Visualize embeddings
    if st.button("Visualize Embeddings"):
        if not st.session_state["vectors"]:
            st.error("Please generate embeddings first!")
        else:
            visualize_embeddings(st.session_state.vectors)

# Main chat interface
st.title("Chat with Your Documents")

# Display chat history
for chat in st.session_state["history"]:
    st.markdown(f"**User:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    st.markdown("---")

# User input for chatbot
user_input = st.text_input("Enter your question:")
if user_input:
    if not st.session_state["vectors"]:
        st.error("Please generate embeddings first!")
    else:
        truncated_docs = truncate_documents_for_token_limit(st.session_state.final_documents)
        document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
            Use the context below to answer the question.
            <context>{context}</context>
            Question: {input}
        """))
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        try:
            with st.spinner("Processing your query..."):
                response = retrieval_chain.invoke({'input': user_input, "context": truncated_docs})
            st.session_state["history"].append({
                "question": user_input,
                "answer": response['answer']
            })
        except Exception as e:
            st.error(f"Error: {e}")
