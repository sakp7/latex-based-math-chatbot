# import os
# import streamlit as st
# import traceback
# # necessary libraries
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.runnables import RunnablePassthrough
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import UnstructuredFileLoader
# from langchain_groq import ChatGroq

# from dotenv import load_dotenv

# load_dotenv()

# def load_document(file_path):
#     """
#     Robust document loading with multiple strategies
#     """
#     try:
#         # Try Unstructured loader for various file types
#         if os.path.exists(file_path):
#             return UnstructuredFileLoader(file_path).load()
        
#         st.error(f"File not found: {file_path}")
#         return []
    
#     except Exception as e:
#         st.error(f"Error loading document: {e}")
#         st.error(traceback.format_exc())
#         return []

# def setup_rag_pipeline(document_path=None):
#     # Allow document path to be configured via Streamlit
#     if document_path is None:
#         document_path = st.file_uploader("Upload a document", type=['pdf', 'txt', 'docx'])
        
#         if document_path is None:
#             st.warning("Please upload a document first")
#             return None, None

#     # If a file is uploaded through Streamlit
#     if hasattr(document_path, 'name'):
#         # Save uploaded file temporarily
#         with open(document_path.name, 'wb') as f:
#             f.write(document_path.getbuffer())
#         document_path = document_path.name

#     # Load documents
#     docs = load_document(document_path)
    
#     if not docs:
#         st.error("No documents could be loaded. Please check the file.")
#         return None, None
    
#     # initializing our embedding model
#     embedding_model = GoogleGenerativeAIEmbeddings(
#         model='models/text-embedding-004', 
#         google_api_key=os.getenv('GOOGLE_API_KEY')
#     )
    
#     # RecusiveCharacter text splitter create small chunks for embedding
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1048, chunk_overlap=150)
#     docs = splitter.split_documents(docs)
    
#     # Creating embedding and storing in FAISS
#     db = FAISS.from_documents(documents=docs, embedding=embedding_model)
#     retriever = db.as_retriever()
    
#     # Initialize LLM
#     llm = ChatGroq(model='llama-3.3-70b-versatile')
    
#     return llm, retriever, docs

# def latex_query_enhancement(llm, query):
#     """
#     Generate a LaTeX-enhanced query using LLM
#     """
#     latex_prompt = ChatPromptTemplate.from_template("""
#     You are a precise mathematical query transformer. Your ONLY task is to:
#     1. Carefully examine the original query
#     2. Convert ALL mathematical expressions to LaTeX notation
#     3. Preserve the EXACT original meaning of the query
#     4. Return ONLY the LaTeX-enhanced query with mathematical precision

#     Rules for Conversion:
#     - Convert all mathematical symbols to LaTeX
#     - Use inline math mode with $ symbols
#     - Ensure mathematical expressions are accurately represented
#     - Do NOT add explanations or additional text

#     Original Query: {query}

#     LaTeX-Enhanced Query:
#     """)
    
#     latex_chain = (
#         latex_prompt 
#         | llm 
#         | StrOutputParser()
#     )
    
#     return latex_chain.invoke({"query": query})

# def rag_pipeline(llm, retriever, query, latex_query):
#     """
#     RAG pipeline with context retrieval and response generation
#     """
#     # Prompt for RAG
#     prompt = ChatPromptTemplate.from_template("""
#     You are an Intelligent AI assistant solving mathematical problems with precision.
    
#     LaTeX-Enhanced Question: {latex_query}
#     Original Question: {question}
    
#     Context: {context}
    
#     Solving Instructions:
#     1. Use the provided context to answer the question
#     2. Break down the solution into clear, numbered steps
#     3. Utilize LaTeX notation for mathematical expressions
#     4. Explain reasoning thoroughly
#     5. If no relevant context exists, clearly state limitations
    
#     Provide a comprehensive, step-by-step solution.
#     """)
    
#     # RAG Chain
#     chain = (
#         {
#             "context": retriever, 
#             "question": RunnablePassthrough(),
#             "latex_query": RunnablePassthrough()
#         }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     return chain.invoke(query)

# def main():
#     st.title("📘 Mathematical Query RAG Assistant")
    
#     # Upload document or use default
#     document_path = st.file_uploader("Upload a document", type=['pdf', 'txt', 'docx'])
    
#     # Setup RAG pipeline
#     if document_path:
#         try:
#             llm, retriever, docs = setup_rag_pipeline(document_path)
            
#             # Create input and submit button
#             query = st.text_input("Enter your mathematical query:")
#             submit_button = st.button("Submit Query")
            
#             if submit_button and query and llm and retriever:
#                 with st.spinner("Processing your query..."):
#                     # Convert query to LaTeX
#                     latex_query = latex_query_enhancement(llm, query)
                    
#                     # Display debugging information
#                     st.subheader("🔍 Query Analysis")
                    
#                     # Original Query
#                     st.markdown("**Original Query:**")
#                     st.code(query)
                    
#                     # LaTeX Query
#                     st.markdown("**LaTeX Enhanced Query:**")
#                     st.code(latex_query)
                    
#                     # Retrieve relevant chunks
#                     retrieved_chunks = retriever.invoke(query)
                    
#                     # Display Retrieved Chunks
#                     st.markdown("**Retrieved Context Chunks:**")
#                     for i, chunk in enumerate(retrieved_chunks, 1):
#                         st.code(f"Chunk {i}: {chunk.page_content}")
                    
#                     # Generate response
#                     response = rag_pipeline(llm, retriever, query, latex_query)
                    
#                     # Display Final Response
#                     st.subheader("🧮 Step-by-Step Solution")
#                     st.markdown(response)
        
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#             st.error(traceback.format_exc())

# if __name__ == "__main__":
#     main()

import os
import streamlit as st
import traceback
# necessary libraries
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def load_documents(file_paths):
    """
    Load multiple documents
    """
    all_docs = []
    for file_path in file_paths:
        try:
            # Try Unstructured loader for various file types
            if os.path.exists(file_path):
                docs = UnstructuredFileLoader(file_path).load()
                all_docs.extend(docs)
            else:
                st.error(f"File not found: {file_path}")
        
        except Exception as e:
            st.error(f"Error loading document {file_path}: {e}")
            st.error(traceback.format_exc())
    
    return all_docs

def create_embedding_store(docs):
    """
    Create embedding store from documents
    """
    # initializing our embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model='models/text-embedding-004', 
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    
    # RecusiveCharacter text splitter create small chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1048, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)
    
    # Creating embedding and storing in FAISS
    db = FAISS.from_documents(documents=split_docs, embedding=embedding_model)
    
    return db

def setup_rag_pipeline():
    # Initialize LLM
    llm = ChatGroq(model='llama-3.3-70b-versatile')
    
    return llm

def latex_query_enhancement(llm, query):
    """
    Generate a LaTeX-enhanced query using LLM
    """
    latex_prompt = ChatPromptTemplate.from_template("""
    You are a precise mathematical query transformer. Your ONLY task is to:
    1. Carefully examine the original query
    2. Convert ALL mathematical expressions to LaTeX notation
    3. Preserve the EXACT original meaning of the query
    4. Return ONLY the LaTeX-enhanced query with mathematical precision

    Rules for Conversion:
    - Convert all mathematical symbols to LaTeX
    - Use inline math mode with $ symbols
    - Ensure mathematical expressions are accurately represented
    - Do NOT add explanations or additional text

    Original Query: {query}

    LaTeX-Enhanced Query:
    """)
    
    latex_chain = (
        latex_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return latex_chain.invoke({"query": query})

def rag_pipeline(llm, retriever, query, latex_query):
    """
    RAG pipeline with context retrieval and response generation
    """
    # Prompt for RAG
    prompt = ChatPromptTemplate.from_template("""
    You are an Intelligent AI assistant solving mathematical problems with precision.
    
    LaTeX-Enhanced Question: {latex_query}
    Original Question: {question}
    
    Context: {context}
    
    Solving Instructions:
    1. Use the provided context to answer the question
    2. Break down the solution into clear, numbered steps
    3. Utilize LaTeX notation for mathematical expressions
    4. Explain reasoning thoroughly
    5. If no relevant context exists, clearly state limitations
    
    Provide a comprehensive, step-by-step solution.
    """)
    
    # RAG Chain
    chain = (
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "latex_query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(query)

def main():
    st.set_page_config(layout="wide")
    st.title("📘 Mathematical Query RAG Assistant")
    
    # Sidebar for PDF uploads
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Upload PDF Documents", 
            type=['pdf','txt'], 
            accept_multiple_files=True
        )
        
        # Embedding creation button
        if st.button("Create Embedding"):
            if uploaded_files:
                # Save uploaded files temporarily
                temp_file_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = uploaded_file.name
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    temp_file_paths.append(temp_path)
                
                # Load documents
                with st.spinner("Loading documents..."):
                    docs = load_documents(temp_file_paths)
                
                # Create embedding store
                with st.spinner("Creating embedding store..."):
                    st.session_state.db = create_embedding_store(docs)
                
                # Success notification
                st.success("Embedding created successfully!")
                st.write(f"Total documents processed: {len(docs)}")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col2:
        # Check if embedding store exists
        if 'db' in st.session_state:
            # Setup RAG pipeline
            llm = setup_rag_pipeline()
            
            # Retriever from existing embedding store
            retriever = st.session_state.db.as_retriever()
            
            # Create input and submit button
            query = st.text_input("Enter your mathematical query:")
            submit_button = st.button("Submit Query")
            
            if submit_button and query:
                with st.spinner("Processing your query..."):
                    # Convert query to LaTeX
                    latex_query = latex_query_enhancement(llm, query)
                    
                    # Display debugging information
                    st.subheader("🔍 Query Analysis")
                    
                    # Original Query
                    st.markdown("**Original Query:**")
                    st.code(query)
                    
                    # LaTeX Query
                    st.markdown("**LaTeX Enhanced Query:**")
                    st.code(latex_query)
                    
                    # Retrieve relevant chunks
                    retrieved_chunks = retriever.invoke(query)
                    
                    # Display Retrieved Chunks
                    st.markdown("**Retrieved Context Chunks:**")
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        st.code(f"Chunk {i}: {chunk.page_content}")
                    
                    # Generate response
                    response = rag_pipeline(llm, retriever, query, latex_query)
                    
                    # Display Final Response
                    st.subheader("🧮 Step-by-Step Solution")
                    st.markdown(response)
        else:
            st.warning("Please upload documents and create embedding in the sidebar")

if __name__ == "__main__":
    main()