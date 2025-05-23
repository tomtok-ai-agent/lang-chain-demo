"""
Example 10: Building a Complete RAG (Retrieval-Augmented Generation) System

This example demonstrates how to build a complete RAG system that combines:
- Document loading and processing
- Vector storage and retrieval
- LLM-based question answering

What it solves:
- Creates a comprehensive system for answering questions based on custom documents
- Demonstrates the full RAG pipeline from document ingestion to answer generation
- Shows how to combine multiple LangChain components into a production-ready system
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Create a sample knowledge base file
    knowledge_base_content = """
    # Company Knowledge Base
    
    ## Products
    Our company offers three main product lines:
    - ProductX: An AI-powered analytics platform for enterprise customers
    - ProductY: A consumer mobile application for personal finance management
    - ProductZ: A cloud-based collaboration tool for small businesses
    
    ## Pricing
    - ProductX: $5,000 per month for up to 100 users, enterprise pricing available
    - ProductY: Free tier available, Premium at $9.99/month
    - ProductZ: $15 per user per month, with annual discount of 20%
    
    ## Support
    Customer support is available 24/7 through:
    - Email: support@example.com
    - Phone: +1-800-555-0123
    - Live chat on our website
    
    ## Company History
    Founded in 2010, our company has grown from a small startup to a global organization with
    offices in New York, London, and Singapore. We received Series A funding in 2012,
    Series B in 2015, and went public in 2020.
    """
    
    # Save the knowledge base to a file
    with open("knowledge_base.txt", "w") as f:
        f.write(knowledge_base_content)
    
    print("Created sample knowledge base file.")
    
    # 1. Load the document
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    
    # 2. Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    print(f"Split into {len(texts)} chunks")
    
    # 3. Create embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)
    
    # 4. Create a question-answering chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever()
    )
    
    # 5. Ask questions
    questions = [
        "What products does the company offer?",
        "How much does ProductY cost?",
        "When was the company founded and where are its offices?",
        "How can customers get support?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        result = qa.invoke({"query": question})
        print(f"Answer: {result['result']}")
    
    # Clean up
    if os.path.exists("knowledge_base.txt"):
        os.remove("knowledge_base.txt")
        print("\nRemoved temporary knowledge base file.")

if __name__ == "__main__":
    main()
