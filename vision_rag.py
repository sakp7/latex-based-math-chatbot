import os
import streamlit as st
import traceback
import base64
from PIL import Image
import io

# necessary libraries
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
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

def image_to_base64(image):
    """
    Convert PIL Image to base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def vision_query_enhancement(image):
    """
    Use Google Gemini Vision Pro to convert image query to text with LaTeX
    """
    # Initialize Google Gemini Vision Pro
    vision_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=os.getenv('GOOGLE_API_KEY')
    )
    
    # Convert image to base64
    base64_image = f"data:image/png;base64,{image_to_base64(image)}"
    
    # Prepare the message for vision processing
    message = HumanMessage(
        content=[
            {"type": "text", "text": """
            Perform a comprehensive OCR on this image with the following guidelines:
            1. Transcribe ALL text in the image precisely
            2. For mathematical equations:
               - Convert to exact LaTeX notation
               - Use $ for inline math and $$ for display math
               - Ensure 100% accurate mathematical representation
            3. Preserve the original context and formatting
            4. If the image contains a mathematical problem, extract the FULL problem statement
            5. If no text is found, return "No text detected"
            """},
            {
                "type": "image_url",
                "image_url": {"url": base64_image},
            },
        ]
    )
    
    # Invoke the vision model
    try:
        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        st.error(f"Vision query enhancement error: {e}")
        return "Error in processing image"
def setup_rag_pipeline():
    # Initialize LLM
    llm = ChatGroq(model="llama3-70b-8192")
    
    return llm

def rag_pipeline(llm, retriever, query, latex_query):
    """
    RAG pipelin`e with context retrieval and response generation
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
            "Upload PDF and Text Documents", 
            type=['pdf', 'txt'], 
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
            
            # Image upload
            uploaded_image = st.file_uploader(
                "Upload Mathematical Query Image", 
                type=['png', 'jpg', 'jpeg']
            )
            
            # Process image and submit
            if uploaded_image:
                # Open the image
                image = Image.open(uploaded_image)
                
                # Display uploaded image
                st.image(image, caption="Uploaded Mathematical Query")
                
                # Submit button
                if st.button("Process Image Query"):
                    with st.spinner("Processing image query..."):
                        # Convert image to text query with LaTeX
                        try:
                            latex_query = vision_query_enhancement(image)
                            

                            # Display query information
                            st.subheader("🔍 Query Analysis")
                            
                            # LaTeX Query
                            st.markdown("**Extracted LaTeX Query:**")
                            st.code(latex_query)
                            
                            # Retrieve relevant chunks
                            retrieved_chunks = retriever.invoke(latex_query)
                            
                            # Display Retrieved Chunks
                            st.markdown("**Retrieved Context Chunks:**")
                            for i, chunk in enumerate(retrieved_chunks, 1):
                                st.code(f"Chunk {i}: {chunk.page_content}")
                            
                            # Generate response
                            response = rag_pipeline(llm, retriever, latex_query, latex_query)
                            
                            # Display Final Response
                            st.subheader("🧮 Step-by-Step Solution")
                            st.markdown(response)
                        
                        except Exception as e:
                            st.error(f"Error processing image: {e}")
                            st.error(traceback.format_exc())
        else:
            st.warning("Please upload documents and create embedding in the sidebar")

if __name__ == "__main__":
    main()