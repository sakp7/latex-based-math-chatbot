
import os
import streamlit as st
import traceback
import base64
import io
from PIL import Image

# Necessary libraries
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        api_key=st.secrets['GOOGLE_API_KEY']
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
    """
    Setup RAG pipeline with Groq LLM
    """
    # Initialize LLM
    llm = ChatGroq(model="llama3-70b-8192",api_key=st.secrets['GROQ_API_KEY'])
    return llm

def retrieve_from_pinecone(query, top_k=3):
    """
    Retrieve relevant documents from Pinecone
    """
    # Initialize embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model='models/text-embedding-004', 
        google_api_key=st.secrets['GOOGLE_API_KEY']
    )
    
    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets['PINECONE_API_KEY'])
    index = pc.Index("mathematical-docs")
    
    # Generate query embedding
    query_embedding = embedding_model.embed_query(query)
    
    # Retrieve from Pinecone
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    
    # Extract and process context
    context_chunks = []
    for result in results['matches']:
        chunk = {
            'text': result['metadata'].get('text', 'No text available'),
            'source': result['metadata'].get('source', 'Unknown source'),
            'score': result['score']
        }
        context_chunks.append(chunk)
    
    return context_chunks

def rag_pipeline(llm, query, latex_query):
    """
    RAG pipeline with context retrieval and response generation
    """
    # Retrieve context from Pinecone
    context_chunks = retrieve_from_pinecone(query)
    
    # Prepare context text for LLM
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    
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
            "context": lambda x: context, 
            "question": lambda x: query,
            "latex_query": lambda x: latex_query
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({}), context_chunks

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

def main():
    st.set_page_config(layout="wide")
    st.title("📘 Mathematical Query RAG Assistant")
    
    # Sidebar with mode selection
    with st.sidebar:
        st.header("🔍 Query Mode")
        query_mode = st.radio(
            "Select Query Mode", 
            ["Text Query", "Image Query"]
        )
    
    # Setup RAG pipeline
    llm = setup_rag_pipeline()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col2:
        if query_mode == "Text Query":
            # Text Input
            query = st.text_input("Enter your mathematical query:")
            submit_button = st.button("Submit Query")
            
            if submit_button and query:
                with st.spinner("Processing your query..."):
                    # Convert query to LaTeX
                    latex_query = latex_query_enhancement(llm, query)
                    
                    # Display query analysis
                    st.subheader("🔍 Query Analysis")
                    st.markdown("**Original Query:**")
                    st.code(query)
                    st.markdown("**LaTeX Enhanced Query:**")
                    st.code(latex_query)
                    
                    # Generate response and get context chunks
                    response, context_chunks = rag_pipeline(llm, query, latex_query)
                    
                    # Display Retrieved Chunks in an Expander
                    with st.expander("🔬 Retrieved Context Chunks"):
                        for i, chunk in enumerate(context_chunks, 1):
                            st.markdown(f"### Chunk {i}")
                            st.markdown(f"**Source:** {chunk['source']}")
                            st.markdown(f"**Relevance Score:** {chunk['score']:.4f}")
                            st.code(chunk['text'])
                    
                    # Display Final Response
                    st.subheader("🧮 Step-by-Step Solution")
                    st.markdown(response)
        
        else:  # Image Query
            # Image upload
            uploaded_image = st.file_uploader(
                "Upload Mathematical Query Image", 
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_image:
                # Open the image
                image = Image.open(uploaded_image)
                
                # Display uploaded image
                st.image(image, caption="Uploaded Mathematical Query")
                
                # Submit button
                if st.button("Process Image Query"):
                    with st.spinner("Processing image query..."):
                        try:
                            # Convert image to text query with LaTeX
                            latex_query = vision_query_enhancement(image)
                            
                            # Display query information
                            st.subheader("🔍 Query Analysis")
                            
                            # LaTeX Query
                            st.markdown("**Extracted LaTeX Query:**")
                            st.code(latex_query)
                            
                            # Generate response and get context chunks
                            response, context_chunks = rag_pipeline(llm, latex_query, latex_query)
                            
                            # Display Retrieved Chunks in an Expander
                            with st.expander("🔬 Retrieved Context Chunks"):
                                for i, chunk in enumerate(context_chunks, 1):
                                    st.markdown(f"### Chunk {i}")
                                    st.markdown(f"**Source:** {chunk['source']}")
                                    st.markdown(f"**Relevance Score:** {chunk['score']:.4f}")
                                    st.code(chunk['text'])
                            
                            # Display Final Response
                            st.subheader("🧮 Step-by-Step Solution")
                            st.markdown(response)
                        
                        except Exception as e:
                            st.error(f"Error processing image: {e}")
                            st.error(traceback.format_exc())

if __name__ == "__main__":
    st.set_page_config(
    page_title="Latext Based RAG",
   
    layout="wide",
    initial_sidebar_state="expanded"
)
    main()
