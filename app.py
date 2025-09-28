"""
AWS Bedrock PDF Chat Application

This application implements an advanced question-answering system using AWS Bedrock services
to enable interactive conversations with PDF documents. The system combines LangChain's document 
processing capabilities with AWS Bedrock's powerful language models to provide accurate and 
contextual responses.

Key Features:
- PDF document ingestion and intelligent chunking for optimal context preservation
- Vector embeddings using AWS Bedrock's Titan model for semantic search
- High-performance similarity search using Facebook AI Similarity Search (FAISS)
- Dual LLM support with AWS Bedrock's Claude and Llama2 models
- Interactive Streamlit-based user interface with real-time processing
- Vector store management for efficient document retrieval

Technical Components:
- LangChain: For document processing, embeddings, and Q&A chains
- AWS Bedrock: For accessing state-of-the-art LLMs and embedding models
- FAISS: For efficient similarity search in high-dimensional space
- Streamlit: For building an interactive web interface

Author: Unknown
Created: Unknown
Last Modified: July 26, 2025
"""

# Standard library imports
import json  # For JSON data handling
import os    # For operating system operations
import sys   # For system-specific parameters and functions

# Web interface framework
import streamlit as st  # For creating the interactive web application

# AWS SDK for interacting with AWS Bedrock service
import boto3

# LangChain components for working with AWS Bedrock
# For text embeddings using AWS models
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock  # For accessing Bedrock LLMs

# Numerical computing library
import numpy as np

# LangChain utilities for text processing
# For splitting documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# For loading PDF documents
from langchain.document_loaders import PyPDFDirectoryLoader

# Vector store for similarity search
from langchain.vectorstores import FAISS  # Facebook AI Similarity Search

# LangChain components for building Q&A systems
# For creating custom prompt templates
from langchain.prompts import PromptTemplate
# For building retrieval-based Q&A systems
from langchain.chains import RetrievalQA

# Initialize AWS Bedrock runtime client for model inference
bedrock = boto3.client(service_name="bedrock-runtime")

# Initialize Bedrock embeddings using Amazon's Titan embedding model
# This creates dense vector representations of text for semantic similarity search
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",  # Using Titan embedding model
    client=bedrock                          # Using the initialized Bedrock client
)


def data_ingestion():
    """
    Ingest and process PDF documents from the 'data' directory.
    
    This function performs the following steps:
    1. Loads all PDF files from the 'data' directory using PyPDFDirectoryLoader
    2. Splits the documents into smaller chunks while preserving context
    3. Uses RecursiveCharacterTextSplitter with optimal chunk size and overlap
    
    Returns:
        list: A list of document chunks, where each chunk contains:
            - page_content: The text content of the chunk
            - metadata: Source document information (filename, page numbers, etc.)
    
    Note:
        The chunk size (10000) and overlap (1000) are chosen to:
        - Maintain enough context for accurate question answering
        - Ensure chunks aren't too large for embedding processing
        - Allow for context overlap to prevent information loss at chunk boundaries
    """
    # Load all PDF files from the data directory
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Split documents into smaller chunks with overlap for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    return docs


def get_vector_store(docs):
    """
    Create and persist a FAISS vector store from document chunks.
    
    This function:
    1. Takes preprocessed document chunks
    2. Uses Bedrock's Titan embeddings to convert text to vectors
    3. Creates a FAISS index for efficient similarity search
    4. Saves the index locally for future use
    
    Args:
        docs (list): List of document chunks from data_ingestion()
                    Each chunk should have text content and metadata.
    
    Note:
        FAISS (Facebook AI Similarity Search) is used because:
        - It's optimized for fast similarity search in high dimensions
        - Supports efficient nearest neighbor search
        - Can handle large-scale document collections
        - Allows for local persistence and reloading
    """
    # Create FAISS vector store using Bedrock's Titan embeddings
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    # Save the vector store locally for future use
    vectorstore_faiss.save_local("faiss_index")


def get_claude_llm():
    """
    Initialize and configure the Anthropic Claude v2 model from AWS Bedrock.
    
    This function sets up Claude with specific parameters:
    - Uses Claude v2:1, known for high-quality text generation
    - Sets max_tokens_to_sample to 512 for controlled response length
    - Configures the model for general Q&A tasks
    
    Returns:
        Bedrock: Configured Claude LLM instance optimized for Q&A tasks.
                The instance is ready to be used in a RetrievalQA chain.
    """
    llm = Bedrock(model_id="anthropic.claude-v2:1",
                  client=bedrock, model_kwargs={'max_tokens_to_sample': 512})
    return llm


def get_llama2_llm():
    """
    Initialize and configure the Meta's Llama2 model from AWS Bedrock.
    
    This function sets up Llama2 with specific parameters:
    - Uses the 8B parameter instruct model variant
    - Sets max generation length to 512 tokens
    - Optimized for instruction-following tasks
    
    Returns:
        Bedrock: Configured Llama2 LLM instance ready for Q&A tasks.
                Provides an alternative to Claude with different strengths.
    """
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0",
                  client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but atleast summarize with 
250 words with detailed explainations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    """
    Generate a response to a query using retrieval-augmented generation (RAG).
    
    This function implements a sophisticated RAG pipeline:
    1. Takes a user query and finds relevant document chunks using FAISS
    2. Retrieves top 3 most similar chunks for context
    3. Combines these chunks with the query in a structured prompt
    4. Uses the specified LLM to generate a contextual response
    
    Args:
        llm: Language model instance (either Claude or Llama2)
        vectorstore_faiss: FAISS vector store with document embeddings
        query (str): User's question to be answered
    
    Returns:
        str: Generated answer that combines:
            - Knowledge from the relevant document chunks
            - The language model's general knowledge
            - Structured according to the prompt template
    
    Note:
        - Uses "stuff" chain type for simpler queries where context fits in one prompt
        - Retrieves top 3 chunks for balanced context vs. relevance
        - Returns source documents for potential future citation needs
    """
    # Create a retrieval QA chain with:
    # - "stuff" chain type (all context is stuffed into a single prompt)
    # - similarity search with top 3 most relevant chunks
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    # Generate answer for the query
    answer = qa({"query": query})
    return answer['result']


def main():
    """
    Main application function implementing the Streamlit web interface.
    
    This function:
    1. Sets up the Streamlit page configuration and layout
    2. Provides a text input for user questions
    3. Offers sidebar controls for vector store management
    4. Implements two separate flows for Claude and Llama2 models
    5. Handles the complete Q&A process including:
        - Vector store loading
        - Model initialization
        - Query processing
        - Response display
    
    The interface allows users to:
    - Ask questions about their PDF documents
    - Choose between Claude and Llama2 for responses
    - Update the vector store with new documents
    - See processing status and success messages
    """
    # Configure the Streamlit page
    st.set_page_config("Chat PDF")

    # Display application header
    st.header("Chat with PDF using AWS Bedrock💁")

    # Get user's question through text input
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()

            # faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()

            # faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
