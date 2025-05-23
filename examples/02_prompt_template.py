"""
Example 2: Using PromptTemplate for Formatting Requests

This example demonstrates how to use PromptTemplate to create reusable prompt templates
with variable slots that can be filled before sending to the model.

What it solves:
- Convenient and safe prompt construction
- Clear definition of input variables and template content
- Avoids problems with string concatenation or missed spaces
- Makes prompts reusable with different data
"""

import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize LLM (OpenAI)
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)
    
    # Define a prompt template with two parameters: style and animal
    template = """Tell a {style} joke about a {animal}."""
    prompt = PromptTemplate(input_variables=["style", "animal"], template=template)
    
    # Use the template with specific values
    formatted_prompt = prompt.format(style="funny", animal="cat")
    print("Formatted prompt:", formatted_prompt)
    
    # Pass the formatted prompt to the model
    response = llm.invoke(formatted_prompt)
    print("\nResponse:")
    print(response)
    
    # Reuse the same template with different values
    formatted_prompt2 = prompt.format(style="sarcastic", animal="dog")
    print("\nFormatted prompt 2:", formatted_prompt2)
    
    # Pass the second formatted prompt to the model
    response2 = llm.invoke(formatted_prompt2)
    print("\nResponse 2:")
    print(response2)

if __name__ == "__main__":
    main()
