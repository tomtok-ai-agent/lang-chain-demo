# LangChain Demo

A repository of example scripts demonstrating the usage of the LangChain library for building applications with Large Language Models (LLMs).

## Overview

This repository contains practical examples of using LangChain, a framework for developing applications powered by language models. Each example demonstrates a specific feature or capability of LangChain, from basic LLM calls to complex retrieval-augmented generation systems.

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tomtok-ai-agent/lang-chain-demo.git
cd lang-chain-demo
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY="your-openai-api-key"
```

## Examples

This repository includes the following examples:

### 1. Simple LLM Call
Basic usage of LangChain to call an OpenAI model.
```bash
python -m examples.01_simple_llm_call
```

### 2. Prompt Templates
Using PromptTemplate to create reusable prompt templates with variable slots.
```bash
python -m examples.02_prompt_template
```

### 3. LLM Chain
Combining a prompt template and a language model into a single chain.
```bash
python -m examples.03_llm_chain
```

### 4. Sequential Chain
Creating multi-step processing chains where the output of one LLM becomes the input to another.
```bash
python -m examples.04_sequential_chain
```

### 5. Conversation Memory
Adding memory to store and retrieve conversation context between chain runs.
```bash
python -m examples.05_conversation_memory
```

### 6. Tools and Agents
Creating a simple tool that can be used by an agent to access external data.
```bash
python -m examples.06_tools_and_agents
```

### 7. Vector Database Integration (FAISS)
Using LangChain with a vector database for semantic search over a collection of texts.
```bash
python -m examples.07_vector_database_faiss
```

### 8. PDF Indexing and Question-Answering
Loading a PDF document, splitting it into chunks, saving to a vector database, and creating a QA chain.
```bash
python -m examples.08_pdf_qa
```

### 9. Agents with Memory and Tools
Combining an agent with memory and tools for a conversational assistant that can remember context and use external tools.
```bash
python -m examples.09_agent_with_memory_and_tools
```

### 10. Complete RAG System
Building a complete Retrieval-Augmented Generation system for answering questions based on custom documents.
```bash
python -m examples.10_complete_rag_system
```

## Notes

- These examples use the latest LangChain version, which has some deprecation warnings. The code has been updated to use the recommended methods where possible.
- The OpenAI models used are the currently supported ones (as of May 2025).
- For production use, consider updating imports to use the specific packages (e.g., `langchain_openai` instead of `langchain_community`) as recommended in the deprecation warnings.

## Documentation

For more detailed information about LangChain, please refer to the [translated document](langchain_english.md) included in this repository, which explains the purpose, architecture, and key components of LangChain.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
