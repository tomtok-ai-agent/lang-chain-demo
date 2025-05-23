"""
Example 4: Sequential Chain for Multi-Step Processing

This example demonstrates how to use SimpleSequentialChain to perform multi-step processing,
where the output of one LLMChain becomes the input to the next.

What it solves:
- Automates context passing between multiple models
- Allows breaking down tasks into steps (decomposition)
- Simplifies complex workflows that require multiple LLM calls
"""

import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize LLM (OpenAI)
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)
    
    # Step 1: Text summarization
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Read the text and briefly summarize the main points in one sentence:\n\"\"\"\n{text}\n\"\"\""
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    
    # Step 2: Translation to another language
    translate_prompt = PromptTemplate(
        input_variables=["summary"],
        template="Translate the following text to Spanish:\n\"\"\"\n{summary}\n\"\"\""
    )
    translate_chain = LLMChain(llm=llm, prompt=translate_prompt)
    
    # Combine into a sequential chain
    overall_chain = SimpleSequentialChain(
        chains=[summary_chain, translate_chain],
        verbose=True
    )
    
    # Sample input text
    input_text = "OpenAI has announced a new model called GPT-4. It exceeds the capabilities of GPT-3.5 and can process images. This new model represents a significant advancement in AI technology and has demonstrated human-level performance on various professional and academic benchmarks."
    
    print("Input text:")
    print(input_text)
    print("\nProcessing through the sequential chain...")
    
    # Run the chain
    final_result = overall_chain.invoke(input_text)
    
    print("\nFinal result (Spanish translation of the summary):")
    print(final_result)

if __name__ == "__main__":
    main()
