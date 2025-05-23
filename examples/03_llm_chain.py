"""
Example 3: LLMChain (Prompt + LLM)

This example demonstrates how to use LLMChain to combine a prompt template and a language model
into a single object that can be called with input variables to get the model's response.

What it solves:
- Encapsulates the "substitute variables into prompt -> call LLM -> get result" step
- Convenient when calling the same prompt with different data
- Automatically handles input/output format conversion
"""

import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize LLM (OpenAI)
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)
    
    # Define a prompt template with two parameters
    template = """Tell a {style} joke about a {animal}."""
    prompt = PromptTemplate(input_variables=["style", "animal"], template=template)
    
    # Create an LLMChain with our model and template
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Now we can call the chain with parameters instead of manual formatting
    result1 = chain.invoke({"style": "sarcastic", "animal": "dog"})
    print("Result 1 (using dictionary):")
    print(result1['text'])
    
    # Alternative way to call with named parameters
    result2 = chain.invoke({"style": "silly", "animal": "penguin"})
    print("\nResult 2 (using named parameters):")
    print(result2['text'])
    
    # We can reuse the same chain with different inputs
    inputs = [
        {"style": "dad", "animal": "cat"},
        {"style": "absurd", "animal": "elephant"}
    ]
    
    # Process multiple inputs
    print("\nProcessing multiple inputs:")
    for i, input_data in enumerate(inputs, 1):
        result = chain.invoke(input_data)
        print(f"\nResult for input {i} ({input_data['style']} joke about {input_data['animal']}):")
        print(result['text'])

if __name__ == "__main__":
    main()
