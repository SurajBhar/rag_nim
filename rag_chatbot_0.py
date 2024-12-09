# Import all dependencies
import os
import time
import json
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

# Initialize session state for embeddings, history, and loader
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "loader" not in st.session_state:
    st.session_state["loader"] = None

# Function to create vector embeddings
def vector_embedding():
    st.session_state.embeddings = NVIDIAEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("./data")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:40]
    )
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

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

# Start developing the Webapp
st.title("Context-aware RAG: NVIDIA NIM")

# Instructions
st.write("""
Welcome to the Context-aware RAG App!
1. Upload your PDF documents.
2. Click the "Document Embedding" button to create the vector database from your documents.
3. Once the embedding is complete, you can optionally summarize the documents or directly ask questions.
4. Ensure your queries or documents do not exceed the token limit of 8192 tokens.
""")

# File uploader
uploaded_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.session_state.loader = PyPDFDirectoryLoader(uploaded_files)
    st.success("PDF documents uploaded successfully!")

# Document embedding
if st.button("Document Embedding"):
    with st.spinner("Embedding documents..."):
        vector_embedding()
    st.success("VectorStoreDB created using NVIDIA embeddings!")

# Optional document summary
if st.session_state.loader and st.button("Summarize Documents"):
    with st.spinner("Summarizing documents..."):
        summary = summarize_documents()
    if summary:
        st.write("Summary of Uploaded Documents:")
        st.write(summary)

# Prompt input
prompt_in = st.text_input("Ask Your Question From PDF Documents")

if prompt_in:
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
                response = retrieval_chain.invoke({'input': prompt_in, "context": truncated_docs})
                response_time = time.process_time() - start
            st.write(f"**Response:** {response['answer']}")
            st.write(f"**Response Time:** {response_time:.2f} seconds")

            # Display similarity search results
            with st.expander("Similarity Search on PDF Documents"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")

            # Save to history
            st.session_state["history"].append({"question": prompt_in, "answer": response["answer"]})
        except Exception as e:
            st.error(f"Error during retrieval or response generation: {e}")

# Search history
if st.session_state["history"]:
    with st.expander("Search History"):
        for item in st.session_state["history"]:
            st.write(f"**Question:** {item['question']}")
            st.write(f"**Answer:** {item['answer']}")
            st.write("---")

# Downloadable results
if st.session_state["history"]:
    if st.button("Download Results"):
        results_data = json.dumps(st.session_state["history"], indent=4)
        st.download_button(
            label="Download Search History as JSON",
            data=results_data,
            file_name="search_history.json",
            mime="application/json"
        )
