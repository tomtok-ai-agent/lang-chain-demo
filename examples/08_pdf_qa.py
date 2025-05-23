"""
Example 8: PDF Indexing and Question-Answering

This example demonstrates how to load a PDF document, split it into chunks,
save it to a vector database, and create a question-answering chain based on it.

What it solves:
- Reduces the barrier to integrating arbitrary files into LLM applications
- Handles PDF parsing and chunking automatically
- Implements the standard RAG (Retrieval-Augmented Generation) pattern
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # For this example, we'll create a simple PDF with some content
    # In a real application, you would use an existing PDF file
    print("This example requires a PDF file. Please create a sample PDF or use an existing one.")
    print("For demonstration purposes, we'll simulate the process with text content.")
    
    # Simulated document content (in a real scenario, this would come from a PDF)
    simulated_docs = [
        "The company reported strong financial results for Q1 2023. Revenue increased by 15% compared to the same period last year.",
        "The board of directors approved a new strategic plan focusing on sustainable growth and digital transformation.",
        "The company plans to expand operations in Asia, with new offices opening in Tokyo and Singapore by the end of the year.",
        "Research and development spending increased by 20%, with a focus on artificial intelligence and machine learning technologies.",
        "The company's customer satisfaction score reached 92%, the highest in the industry according to independent surveys."
    ]
    
    print("\nSimulating document loading and processing...")
    
    # In a real application, you would use:
    # loader = PyPDFLoader("your_document.pdf")
    # documents = loader.load()
    
    # Split documents into chunks for better search
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(simulated_docs)
    
    print(f"Document split into {len(docs)} chunks")
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings)
    
    # Create a QA chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" method puts all retrieved docs into the prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )
    
    # Ask questions
    questions = [
        "What were the financial results for Q1?",
        "What are the company's expansion plans?",
        "What is the customer satisfaction score?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")

if __name__ == "__main__":
    main()
