"""
PDF CHAT CLIENT APPLICATION
--------------------------
An elegant RAG (Retrieval Augmented Generation) system that allows users to chat with
their PDF documents using AWS Bedrock and FAISS vector database.

This application loads pre-indexed PDF documents from S3, processes user questions,
and returns contextually relevant answers using AWS Bedrock Nova model.
"""

import boto3
import streamlit as st
import os
import uuid
import time
from typing import Dict, Any

# Configure page settings for a professional appearance
os.environ["AWS_REGION"] = "eu-central-1"

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance UI appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize S3 client for fetching vector stores
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Import necessary LangChain components with updated imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock as BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Initialize AWS Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="eu-central-1")

# Configure embeddings model with proper imports
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", 
    client=bedrock_client
)

# Define path for temporary storage
folder_path = "/tmp/"

def get_unique_id() -> str:
    """
    Generate a unique identifier for request tracking.
    
    Returns:
        str: UUID string for the current session
    """
    return str(uuid.uuid4())

def load_index() -> None:
    """
    Download FAISS index files from S3 bucket to local storage.
    
    This function retrieves both the .faiss and .pkl files required
    to reconstruct the vector database locally.
    """
    try:
        with st.status("Downloading vector index from S3...") as status:
            # Download the FAISS index file
            s3_client.download_file(
                Bucket=BUCKET_NAME, 
                Key="my_faiss.faiss", 
                Filename=f"{folder_path}my_faiss.faiss"
            )
            status.update(label="Downloaded FAISS index", state="running", expanded=True)
            
            # Download the pickle file with metadata
            s3_client.download_file(
                Bucket=BUCKET_NAME, 
                Key="my_faiss.pkl", 
                Filename=f"{folder_path}my_faiss.pkl"
            )
            status.update(label="Vector index ready!", state="complete")
    except Exception as e:
        st.error(f"Error downloading index: {str(e)}")
        st.stop()

def get_llm() -> BedrockLLM:
    """
    Initialize and configure the Bedrock LLM.
    
    Returns:
        BedrockLLM: Configured Bedrock LLM instance
    """
    # Use llama3-2-1b-instruct-v1 model with appropriate parameters
    llm = BedrockLLM(
        model_id="eu.meta.llama3-2-1b-instruct-v1:0",
        client=bedrock_client,
        model_kwargs={
            "temperature": 0.2,
            "maxTokens": 512,  # Nova uses camelCase parameter names
            "topP": 0.9,
            # Add any other Nova-specific parameters here
        }
    )
    return llm

def get_response(llm: BedrockLLM, vectorstore: FAISS, question: str) -> str:
    """
    Process user question through RAG pipeline to generate contextual answer.
    
    Args:
        llm: The language model to use for response generation
        vectorstore: FAISS vector store containing document embeddings
        question: User's question text
        
    Returns:
        str: Generated answer based on relevant context
    """
    # Define standard prompt template for the Nova model
    # Using a format that works with standard completion models
    prompt_template = """
    Instructions: You are an expert document assistant. Please use only the information in the provided context 
    to answer the question accurately and concisely.
    
    If the context doesn't contain the information needed to answer the question, respond with:
    "I don't have enough information in the document to answer this question."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""

    # Create standard prompt with input variables
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # Configure retrieval-based QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Combines all retrieved documents into one context
        retriever=vectorstore.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={"k": 5}     # Return top 5 most relevant chunks
        ),
        return_source_documents=True,  # Include source documents in response
        chain_type_kwargs={"prompt": PROMPT}  # Use our custom prompt
    )
    
    # Get answer from the chain - using invoke() instead of __call__
    answer = qa.invoke({"query": question})
    return answer['result']

def main():
    """
    Main application function with enhanced Streamlit UI.
    Handles index loading, user interaction, and response generation.
    """
    # Display application header with custom styling
    st.markdown('<div class="main-header">PDF Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Powered by AWS Bedrock & RAG Architecture</div>', unsafe_allow_html=True)
    
    # Add sidebar information
    with st.sidebar:
        st.image("https://placeholder.pics/svg/300x100/DEDEDE/555555/AWS%20Bedrock", width=300)
        st.subheader("About this app")
        st.info("""
        This application uses Retrieval Augmented Generation (RAG) to chat with your PDF documents.
        
        The system:
        1. Loads pre-indexed document vectors from S3
        2. Matches your question with relevant document sections
        3. Uses Amazon Nova to generate precise answers based on document content
        """)
        st.subheader("Session Info")
        session_id = get_unique_id()
        st.code(f"Session ID: {session_id}")
    
    # Main area - Load index and prepare vector store
    with st.spinner("Preparing document index..."):
        load_index()
        
        # For debugging - display files in temporary directory
        with st.expander("System Debug Info", expanded=False):
            dir_list = os.listdir(folder_path)
            st.write(f"Files in {folder_path}:")
            st.code("\n".join(dir_list))
    
    # Load FAISS index with embeddings
    try:
        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("📚 Document knowledge base successfully loaded!")
    except Exception as e:
        st.error(f"Failed to load index: {str(e)}")
        st.stop()
    
    # User input section
    st.subheader("Ask me anything about your document")
    question = st.text_input("Your question:", placeholder="E.g., What are the main points discussed in the document?")
    
    col1, col2 = st.columns([1, 6])
    with col1:
        submit_button = st.button("Ask 🔍")
    
    # Chat history container
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Process question when button is clicked
    if submit_button and question:
        st.session_state.chat_history.append({"question": question, "answer": None})
        
        with st.spinner("Analyzing document and generating answer..."):
            # Get LLM instance
            llm = get_llm()
            
            # Track response time for metrics
            start_time = time.time()
            
            # Get response from RAG pipeline
            response = get_response(llm, faiss_index, question)
            
            # Calculate and log response time
            response_time = time.time() - start_time
            
            # Update the latest question with its answer
            st.session_state.chat_history[-1]["answer"] = response
            st.session_state.chat_history[-1]["time"] = f"{response_time:.2f}s"
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for i, exchange in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q: {exchange['question']}**")
            if exchange["answer"]:
                st.markdown(f"{exchange['answer']}")
                if "time" in exchange:
                    st.caption(f"Response time: {exchange['time']}")
            st.divider()
    
    # Add footer
    st.markdown("---")
    st.caption("PDF Chat Assistant © 2025 ")

if __name__ == "__main__":
    main()