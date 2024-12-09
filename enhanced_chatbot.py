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

# Initialize session state for embeddings, history, and loader
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
        st.session_state.docs[:40]
    )
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Function to visualize embeddings
def visualize_embeddings(vectors):
    # Retrieve embeddings directly from the FAISS index
    embeddings = np.array([vectors.index.reconstruct(i) for i in range(vectors.index.ntotal)])
    
    # Perform t-SNE on the embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot the t-SNE results
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    plt.title("t-SNE Visualization of Document Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(plt)

# Function to summarize documents
def summarize_documents():
    from langchain.chains.summarize import load_summarize_chain
    summary_chain = load_summarize_chain(llm, chain_type="stuff")
    summary_text = ""
    try:
        summary_text = summary_chain.invoke({"input_documents": st.session_state.final_documents})
    except Exception as e:
        st.error(f"Error during summarization: {e}")
    return summary_text

# Function to check token limits and truncate if necessary
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

    # Document embedding
    if st.button("Document Embedding"):
        with st.spinner("Embedding documents..."):
            vector_embedding()
        st.success("VectorStoreDB created using NVIDIA embeddings!")

    # Document summary
    if st.button("Summarize Documents"):
        if not st.session_state["vectors"]:
            st.error("Please create embeddings first by clicking the 'Document Embedding' button.")
        else:
            with st.spinner("Summarizing documents..."):
                summary = summarize_documents()
            if summary:
                st.write("Summary of Uploaded Documents:")
                st.write(summary)

    # Embedding visualization
    if st.button("Visualize Embeddings"):
        if not st.session_state["vectors"]:
            st.error("Please create embeddings first by clicking the 'Document Embedding' button.")
        else:
            visualize_embeddings(st.session_state.vectors)

    # Downloadable results
    if st.session_state["history"]:
        if st.button("Download Search History"):
            results_data = json.dumps(st.session_state["history"], indent=4)
            st.download_button(
                label="Download Search History as JSON",
                data=results_data,
                file_name="search_history.json",
                mime="application/json"
            )

# Right-hand side panel for instructions
with st.container():
    st.sidebar.markdown(
        """
        ### How to Use:
        1. **Upload Documents**: Optionally upload PDFs or use default data from the `data` directory.
        2. **Create Embeddings**: Click "Document Embedding" to process documents.
        3. **Optional**: Summarize documents or visualize embeddings.
        4. **Chat**: Enter your queries in the chat box below to interact with the processed documents.
        """
    )

# Main chat interface
st.title("Chat with Your Documents")

# Display chat history
for chat in st.session_state["history"]:
    st.markdown(f"**User:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    st.markdown("---")

# User input
user_input = st.text_input("Enter your question:", key="user_input")
if user_input:
    if not st.session_state["vectors"]:
        st.error("Please create embeddings first by clicking the 'Document Embedding' button.")
    else:
        # Check and truncate documents for token limit
        truncated_docs = truncate_documents_for_token_limit(st.session_state.final_documents)
        document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Questions:{input}
        """))
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Generate response
        try:
            with st.spinner("Processing your query..."):
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_input, "context": truncated_docs})
                response_time = time.process_time() - start
            st.session_state["history"].append({
                "question": user_input,
                "answer": response['answer']
            })
            # Update session to refresh UI
            st.session_state["trigger_refresh"] = time.time()
        except Exception as e:
            st.error(f"Error during retrieval or response generation: {e}")
