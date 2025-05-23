"""
Example 5: Adding Memory for Storing Conversation Context

This example demonstrates how to use Memory to store and retrieve conversation context
between chain runs, allowing the model to remember previous interactions.

What it solves:
- Automatically adds previous messages to the model's prompt
- Manages conversation history without manual tracking
- Essential for chatbots and virtual assistants where responses should build on previous exchanges
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize chat model (better suited for conversation)
    chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    # Create a memory object - buffer memory (saves all messages)
    memory = ConversationBufferMemory(return_messages=True)
    
    # Create a conversation chain with this memory
    conversation = ConversationChain(
        llm=chat_llm, 
        memory=memory, 
        verbose=True
    )
    
    # Simulate a dialogue
    print("Starting conversation with the AI assistant...\n")
    
    # First message
    response1 = conversation.invoke({"input": "Hello! What's your name?"})
    print("User: Hello! What's your name?")
    print(f"AI: {response1['response']}\n")
    
    # Second message
    response2 = conversation.invoke({"input": "What is 2+2?"})
    print("User: What is 2+2?")
    print(f"AI: {response2['response']}\n")
    
    # Third message - testing memory
    response3 = conversation.invoke({"input": "What did I ask you in my first message?"})
    print("User: What did I ask you in my first message?")
    print(f"AI: {response3['response']}\n")
    
    # We can examine the memory to see what's stored
    print("Memory contents:")
    print(memory.buffer)

if __name__ == "__main__":
    main()
