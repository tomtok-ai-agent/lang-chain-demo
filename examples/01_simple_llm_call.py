"""
Example 1: Simple LLM Call via LangChain

This example demonstrates the most basic operation - calling a language model to generate text.
It shows how to initialize an OpenAI model and make a simple call with a prompt.

What it solves:
- Simplifies addressing the model
- Allows setting parameters once during initialization
- Abstracts API details for easy model switching
"""

import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Set your OpenAI API key
    # You can set it as an environment variable or directly here
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Initialize LLM (OpenAI) - using gpt-3.5-turbo-instruct as text-davinci-003 is deprecated
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)  # temperature determines creativity
    
    # Call the model with a simple prompt
    prompt = "Write a short poem about stars"
    response = llm.invoke(prompt)  # Using invoke() instead of __call__
    
    print("Prompt:", prompt)
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
