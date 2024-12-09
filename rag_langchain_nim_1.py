# Import all dependencies
import os
import time
from dotenv import load_dotenv
# Import streamlit for Web app
import streamlit as st
# Langchain Specific Imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables: API Key
load_dotenv()
os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")


# Create LLM model for inference from NVIDIA-NIM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Create a function to create vector embeddings from text chunks
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./data") ## Data Ingestion: read PDF Files
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:40]) #splitting + Top 40 documents
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #OpenAI Vector embeddings

# Start developing the Webapp
st.title("Context-aware RAG: NVIDIA NIM")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)


prompt_in=st.text_input("Ask Your Question From PDF Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Created VectorStoreDB using Nvidia Embedding")

if prompt_in:
    document_chain=create_stuff_documents_chain(llm,prompt) # Creating document chain
    retriever=st.session_state.vectors.as_retriever() # Interface to retrieve all the data from vector store
    retrieval_chain=create_retrieval_chain(retriever,document_chain) # Create Retrieval chain
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt_in})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    # With a Streamlit Expander
    with st.expander("Similarity Search on PDF Documents"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
