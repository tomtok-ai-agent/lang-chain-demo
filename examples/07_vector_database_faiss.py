"""
Example 7: Integration with Vector Database (FAISS)

This example demonstrates how to use LangChain with a vector database (FAISS)
for semantic search over a collection of texts.

What it solves:
- Simplifies working with embeddings and vector databases
- Provides a unified interface for different vector stores
- Enables semantic search based on meaning rather than keywords
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Sample database of texts (countries and their capitals)
    texts = [
        "France - capital city is Paris.",
        "Germany - capital city is Berlin.",
        "Italy - capital city is Rome.",
        "Spain - capital city is Madrid.",
        "United Kingdom - capital city is London.",
        "Japan - capital city is Tokyo.",
        "Australia - capital city is Canberra.",
        "Brazil - capital city is Brasília."
    ]
    
    print("Creating vector embeddings for our text database...")
    
    # Initialize the embeddings model (OpenAI Embeddings)
    embeddings = OpenAIEmbeddings()  # uses text-embedding-ada-002 by default
    
    # Create a FAISS vector store from the list of texts
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    
    # Now we can perform semantic search
    queries = [
        "What city is the capital of Germany?",
        "Tell me about the capital of Japan",
        "Which European country has Madrid as its capital?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Perform similarity search
        docs = vector_store.similarity_search(query, k=1)
        
        print(f"Most relevant document: {docs[0].page_content}")
        
        # In a real application, you would typically:
        # 1. Take this relevant document
        # 2. Use it as context in a prompt to an LLM
        # 3. Generate a natural language response
    
    # Example of saving and loading the index (commented out)
    # vector_store.save_local("faiss_index")
    # loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)

if __name__ == "__main__":
    main()
