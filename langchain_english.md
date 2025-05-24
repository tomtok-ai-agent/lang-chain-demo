# LangChain: Purpose, Architecture, and Key Components

LangChain is a popular open-source framework for developing applications based on large language models (LLMs). It provides high-level abstractions and ready-made components that allow building complex chains of interaction with LLMs instead of one-time model calls. Simply put, LangChain helps not only to call the model via API but also to connect external data and integrate tools (such as knowledge bases, search, calculator, etc.) to expand the model's capabilities. Below we will examine why such tools are needed, the architecture of LangChain, key components, alternatives (LlamaIndex, Haystack, Semantic Kernel, etc.), and provide 10 Python code examples demonstrating the use of LangChain – from simple calls to building a chatbot with memory and an external knowledge base.

## Why LangChain and Similar Frameworks Are Needed

Developing LLM applications "from scratch" involves a number of repetitive tasks and challenges. Without frameworks like LangChain, developers would have to manually solve the following problems:

- **Prompt formation and response processing**. When directly using the OpenAI API or HuggingFace library, you need to format the request (prompt) for the model and parse its output yourself. In complex scenarios (e.g., multi-step dialogues, action chains), it's easy to get confused in message templates and roles (system, user, assistant).
- **Context and memory management**. Models have a limited context window, so to maintain a conversation, you need to send the chat history each time. Without specialized tools, the developer collects the latest replies or dialogue summary themselves. This is complex and prone to information loss or context overflow.
- **Access to data and knowledge**. LLMs don't know your private data or documents. You need to connect external knowledge sources: files, databases, web search. Without frameworks, this requires implementation of: document loading, splitting into fragments, obtaining embeddings, searching through them, and embedding the found text into the prompt. All these stages need to be programmed and debugged manually.
- **Integration of tools and actions**. Often, the model should not only generate text but also perform actions: make an HTTP request to an API, perform a calculation, get the current date, call an external function. Without an LLM-based agent, the programmer would have to hardcode the logic: if the user asks X – call tool Y. This is difficult to scale to arbitrary requests.
- **Multi-step chains and planning**. Complex tasks may require several processing steps: for example, first extract information, then generate a response. Without a framework, each step is a separate model call with manual result passing, as well as deciding when and what to call.
- **Logging, monitoring, debugging**. In complex LLM processes, it's important to see intermediate steps: what prompt was formed, what was found in the database, which tool was called. Without ready-made solutions, you need to manually implement logs at each stage.

LangChain solves these problems by providing ready-made modules and patterns for the above tasks. In fact, such frameworks are the "plumbing" for LLM applications, connecting models, data, and tools. They significantly reduce the amount of code that needs to be written and offer already debugged solutions from experts. With LangChain, developers can focus on the logic of their application, combining ready-made blocks, instead of reinventing the wheel. As a result, reliability increases (components are tested by the community) and development accelerates.

Without such tools, development would be more labor-intensive and error-prone. For example, implementing a simple chatbot with access to documentation would require independently: writing a document parser, embedding search by embeddings, constructing a prompt with the found context, storing dialogue history, etc. With LangChain, many of these tasks are solved with a few lines using standard classes.

## LangChain Architecture and Key Components

LangChain is designed modularly. The official documentation highlights six main modules of the framework:

- **Model I/O (Models)**: interfaces for interacting with LLMs and chat models. Both commercial APIs (OpenAI, Anthropic, etc.) and open-source models (HuggingFace Transformers) are supported. LangChain introduces a unified calling standard: models have a predict method (or simply calling as a function) that takes a string or messages and returns a result. For example, the OpenAI class wraps the OpenAI API call, and HuggingFaceHub wraps models from HuggingFace. There are two types of models – LLM (takes text, returns text) and ChatModel (takes a list of messages with roles, returns a message). This layer simplifies model switching – you can replace GPT-3.5 with GPT-4 or local LLaMA, with almost no changes to the rest of the code.

- **Data Connection (Data)**: components for accessing the application's own data. This includes Document Loaders – a set of classes for loading documents of different formats (PDF, HTML, TXT, DOCX, databases, etc.), as well as Text Splitters for breaking long texts into reasonable fragments (to fit into the model's context). This category also includes Embedding models (generating vector representations of text) and Vector Stores – vector databases for semantic search. LangChain supports many repositories: from simple local FAISS or Chroma to cloud solutions like Pinecone, Weaviate, Qdrant, and even search engines like ElasticSearch with vector plugins. These integrations make it easy to add a knowledge search stage to your application: load documents, create embeddings, save to storage, and then find relevant pieces based on user queries. This implements the Retrieval-Augmented Generation (RAG) approach. LangChain provides ready-made chains for RAG (e.g., RetrievalQA), which automate the entire process: search the vector database for the question, add the found text to the prompt, and get the final answer from the LLM.

- **Chains**: a mechanism for sequential execution of multiple operations. A Chain is essentially a sequence of calls (models or other components), where the output of one step can be the input of another. Chains allow breaking down a complex task into several stages. LangChain includes a number of standard chains:
  - **LLMChain** – the simplest chain, consisting of a Prompt template and an LLM. It substitutes input variables into the template and calls the model.
  - **Sequential Chain** – a chain where steps are performed one after another: the output of the first step is passed to the input of the second, and so on. There is a simplified version SimpleSequentialChain (sequentially calls a list of LLMChains) and a more flexible SequentialChain with explicit indication of which fields of which step go where.
  - **Router Chain** – a router chain: uses a special model to choose which of several chains to run for a given input (useful when different types of requests in an application require different processing).
  - **Transform Chain** – a step that performs an arbitrary Python function for data transformation (e.g., post-processing text, filtering).

Chains can be nested within each other, forming processing graphs. In essence, Chains are the "Swiss Army knife" for building pipelines with LLMs: instead of a monolithic prompt, you can explicitly distribute logic across steps. Without LangChain, you would have to manually call the model at each step and pass state between calls, monitoring data formatting. With LangChain, chains do this automatically.

- **Agents**: one of the most advanced features of LangChain. An agent is a component that decides for itself what actions to take to achieve a set goal. Inside, the agent uses an LLM as a "brain" which, based on instructions and current context, generates "Thoughts" and "Actions". Actions are formatted as calls to one of the available tools. After performing an action, the agent receives an Observation (result) and again decides what to do next. Thus, the sequence of steps is not hardcoded (as in a Chain) but is determined dynamically by the model itself in dialogue mode.

LangChain provides basic implementations of agents (Reactive, Conversational, MRKL, etc.), as well as integrations with numerous tools. Tools are simply functions (or wrappers around external APIs) that the agent can call through the Action Input/Output format. Examples of tools: web search, knowledge bases, calculator, Python script, document database search, any of your APIs. The more tools an agent has, the wider the range of tasks it can perform. However, the agent itself is an LLM, so it needs to be provided with a description of what tools are available and when to use them. The developer defines the list of tools and the basic prompt ("solve the task by choosing from these actions..."). Then the model itself plans the chain of calls. For example, if you ask such an agent: "What's the weather in Paris now and translate the description into Spanish", the agent could first call the weather tool for Paris, get a response, then call the translator. Without LangChain, such behavior would have to be coded with conditional operators or by writing your own parser in natural language → actions, which is extremely non-trivial. LangChain implements the well-known ReAct (Reason+Act) paradigm with minimal code.

The difference between a chain and an agent: a chain is a fixed scenario (defined by the programmer), an agent is a scenario developed by the model itself on the fly. An agent is more flexible but more complex and expensive (each "thought step" is a separate request to the LLM). In practice, if the task is structured and predictable, Chains are used; if the list of actions is not known in advance – Agents.

- **Memory**: a module for storing and restoring state between chain/agent runs. Most often, memory refers to dialogue history that needs to be included in the prompt so that the model remembers the conversation context. LangChain provides different memory implementations:
  - **Buffer Memory** – the simplest option: stores a complete buffer of all messages. With each new request, the entire history (or N last messages) is added to the prompt.
  - **Buffer Window Memory** – a limited window of K last messages (useful to avoid exceeding the context).
  - **Conversation Summary Memory** – stores not the entire history, but a dynamically updated summary of the conversation. After each reply, the old summary can be re-summarized taking into account the new message, allowing to preserve a long context in a compressed form.
  - **Knowledge-graph / Entity Memory** – advanced options that extract facts, entities from the dialogue and store them in a structured form (e.g., knowledge graph, dictionary). This allows the model to remember specific details (names, preferences) without repeating the entire dialogue.

Under the hood, memory is implemented simply as a message store and a method that returns variables for the prompt (e.g., the history variable for substitution). When a chain/agent is launched, it combines the current task + memory content and passes it to the LLM. Without LangChain, the developer would manually assemble a string like "History: ... \n New question: ..." at each step. With memory, it's enough to specify which memory to use – the rest is automated. Memory is especially important for chatbots and voice assistants, where sequential interaction with the user is required.

- **Callbacks**: a system of logging and hooks. LangChain allows subscribing to events of chain/agent execution – start of model call, receiving a response, using a tool, etc. Through callbacks, you can implement step-by-step output of the process to the interface (streaming), collect telemetry, or visualize the agent's reasoning process. This is useful for debugging and monitoring LLM applications. For example, LangChain easily integrates with LangSmith – a platform for debugging and evaluating LLM prompts, or simply outputs intermediate steps to the console with verbose=True. Callbacks help understand exactly how the model arrived at the answer.

In total, the LangChain architecture covers the full cycle of an LLM application: from connecting to the model and data, to complex decision-making logic, storing context, and integration with the external world. The developer can use as many components as the task requires. For example, for a simple Q&A scenario, a Chain (RetrievalQA) without explicit agents is sufficient, and for a conversation assistant with web search – an agent with tools and memory.

## Alternatives to LangChain and Their Applications

The ecosystem of tools for LLMs is rapidly developing, and LangChain is not the only choice. Let's consider several popular alternatives, their features, and application scenarios:

### LlamaIndex (formerly GPT Index)

LlamaIndex is a framework focusing on connecting your own data to language models. Simply put, LlamaIndex is designed for building context-enriched LLM applications where you combine your data with the power of the model. Typical cases: a chatbot answering questions about your documents; extracting information from a knowledge base; analysis and summary of large texts.

The key idea of LlamaIndex is creating indices on top of data. It provides:
- **Data connectors**: connectors for loading data from different sources (files, databases, APIs).
- **Indices**: structures for storing data representations (e.g., vectors). Several types of indices allow efficiently extracting the needed context – from a simple list of documents to complex summary trees.
- **Query/Response engine**: a layer that takes a user's question, finds the necessary pieces of data (through the index), and forms an answer with their help using an LLM.
- **Tools and Agents**: LlamaIndex also supports the agent approach – for example, an agent capable of traversing different indices in several steps or performing actions based on data.
- **Evaluations**: built-in metrics and logging for tracking the quality of model answers based on your data.

Simply put, LlamaIndex can be thought of as a specialized tool for Retrieval-Augmented Generation (RAG). If LangChain provides a "general toolbox", then LlamaIndex is more focused on connecting text and models. These two frameworks are often used together: for example, LlamaIndex can build a complex index on top of a corporate document base, and a LangChain agent can use it as one of its tools.

When to use: if your task is primarily questions and answers about documents, chat with embedded knowledge, fact extraction – LlamaIndex provides a convenient interface for building indices and executing queries. It is well suited for scenarios where a lot of logic around data is needed (e.g., different types of indexing, complex search chains). The LlamaIndex community is smaller than LangChain's, but is actively growing.

### Haystack

Haystack is an open framework from deepset, originally created for building question-answering systems based on document reading (supporting models like BERT). Now Haystack has evolved into a universal tool for search-generative applications. It positions itself as a framework for RAG, chatbots, agents, and semantic search across large document collections.

Haystack features:
- **Components and pipelines**. Haystack offers a rich library of components: retrievers (BM25, DPR, vector search), rerankers, readers based on LLM or classical models, query processors, connectors to data sources. Components can be connected in a pipeline – essentially an execution graph (can even be non-linear, DAGs are supported). For example, a pipeline: "get question -> extract candidates from ElasticSearch -> pass through re-rank model -> generate answer with LLM based on top-5 documents".
- **Integrations**. Haystack integrates with popular ML services: Hugging Face Hub (for models), OpenAI API, Cohere, Azure, SageMaker, as well as data stores: OpenSearch/Elasticsearch, Pinecone, Qdrant, Weaviate, etc. There are also tools for data loading, quality evaluation (eval), monitoring, etc. – covering the full cycle from data to deployment.
- **deepset Cloud and Studio**. For corporate use, there is a cloud platform based on Haystack (deepset Cloud) and the visual environment deepset Studio for designing and debugging pipelines.

When to use: Haystack is historically strong in text search scenarios and classical QA (when you need to find a specific answer in documents). If you already have a search index (e.g., Elasticsearch) and need to add generative answers – Haystack will be a natural choice. It is also attractive for production systems where flexibility and reliability are needed: you can precisely configure how a request passes through the system, replace components with your own. At the same time, the entry threshold may be slightly higher than with LangChain, due to the need to understand the concept of pipelines. Haystack is fully in Python (LangChain also has a JS/TS version, and SK – in .NET). An important difference is the "pipeline vs. agent" approach: Haystack encourages explicitly setting the sequence of steps, while LangChain allows delegating this to the LLM (although Haystack now also supports agent mode). In the end, Haystack is an excellent choice for search systems, FAQ bots, enterprise QA, where control over data and processing stages is important.

### Semantic Kernel

Semantic Kernel (SK) is an open SDK from Microsoft for creating LLM applications, especially popular in the .NET ecosystem (C#). It can be viewed as Microsoft's analog to LangChain, with some additional capabilities. Semantic Kernel is "the glue connecting LLMs with data and code", acting as an orchestrator in applications like Copilot.

Key concepts of SK:
- **Skills and Plugins**. In Semantic Kernel, functions that the model can perform are combined into Plugins. A plugin is a set of methods (e.g., a "Calendar" plugin with methods to add an event, find free time, etc.). The model can call these functions, similar to how a LangChain agent calls tools. The developer describes the plugin and registers it in the "kernel".
- **Prompts and Planner**. SK actively uses prompt templates, which are stored directly as files (you can write long prompts with instructions and variables). A unique feature is the planner: Semantic Kernel can use the model to independently compose a plan of sequential steps to fulfill a user's request. That is, given a task, SK can generate which skills/functions need to be called and in what order to achieve the goal. This plan is then executed. This is the next step compared to simple function calling – the model not only chooses from actions but also combines them into a solution to the problem.
- **Context and memory**. SK provides similar capabilities to LangChain for storing histories, as well as a separate concept of Context Variables – a dictionary of data available in the context when executing plans (e.g., results of previous steps).
- **Language support**. Initially, SK was written in C# (for .NET developers), but there is also a Python version. However, not all functions are equivalent in all languages at the moment.

When to use: Semantic Kernel is especially attractive for C#/.NET developers wanting to integrate LLMs into applications on the Microsoft platform. It fits well into the enterprise stack, supports Azure OpenAI, and is available as a NuGet package. The SK planner is convenient for dynamically changing tasks. If you need a hybrid of "LLM + classical code" – SK allows describing part of the logic as regular functions (plugins) and part trusting to the model. For example, your system may have a number of optional actions (database query, calculation, API call), and the model in Semantic Kernel will decide in what order to pull them depending on user input. At the same time, similar capabilities can be implemented in Python with LangChain (through agents, OpenAI functions with planning plugins, etc.), but SK offers a ready-made planning mechanism "out of the box".

Besides the above, there are other tools:
- **Hugging Face Transformers** – a popular library for working with the models themselves (including basic pipelines for question-answering, summarization, etc.). This is more of a low-level toolkit for models than an orchestration framework.
- **OpenAI Function Calling / Agents API** – recent OpenAI updates allow connecting functions to ChatGPT, essentially turning it into an agent. This is an alternative path: not using LangChain, but directly using the API of models with functions. However, then part of the responsibilities (parsing requests, managing context) will fall on you. LangChain can also ease this task, as it knows how to work with function calling.
- **Dust, Flowise, GPT Engineer, etc.** – many new projects try to simplify the creation of LLM applications, often through a graphical interface or high-level templates. For example, Flowise is a no-code visual constructor based on LangChain components.

It's important to note: in terms of functionality, LangChain, LlamaIndex, Haystack, Semantic Kernel largely overlap. They all solve the common task – simplifying the integration of LLMs into applications. The difference is in emphasis: LangChain – maximum flexibility and community templates; LlamaIndex – working with user data and indexing; Haystack – powerful search pipelines and control; Semantic Kernel – integration with the .NET ecosystem and planning. The choice often comes down to convenience and environment. All the listed frameworks are open and free, so you can try each on a pilot project and see what better fits your requirements.

## Examples of Using LangChain (Python)

Now let's move to the practical part. Below are 10 Python code examples demonstrating various capabilities of LangChain – from simple operations to a complex agent. The examples are designed for developers familiar with Python (or C#) and basic LLM concepts, but without deep experience with such frameworks. Each example is accompanied by an explanation: why it's needed, what problem it solves, and how the solution would look without LangChain.

Before starting, make sure you have installed the langchain package (e.g., `pip install langchain openai faiss-cpu chromadb`), and have an OpenAI API key (it will be needed to call OpenAI models). The code below assumes the presence of the OPENAI_API_KEY environment variable, or you can explicitly specify the key when initializing the model.

### 1. Simple LLM Call via LangChain

The most basic operation is calling a model to generate text. Without frameworks, you could use `openai.Completion.create(...)` or a similar library method directly. In LangChain, this is done through the LLM class corresponding to the desired model/provider. For example, OpenAI is a wrapper around the OpenAI Completion API.

What it solves: simplifies addressing the model. You can set parameters (temperature, maximum length, etc.) once during initialization, and then call the model as a function. This also abstracts API details – it's easy to switch models, for example, to ChatOpenAI (chat interface) or HuggingFaceHub without changing the rest of the code.

Without LangChain: direct call via HTTP or SDK: you need to form a request dictionary, specify the endpoint/model version, process the JSON response. LangChain encapsulates this routine.

```python
# Initialize LLM (OpenAI) - gpt-3.5-turbo-instruct model for simplicity.
from langchain_community.llms import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7)  # temperature determines creativity

# Call the model with a simple prompt:
prompt = "Write a short poem about stars"
response = llm.invoke(prompt)

print(response)
```

In this code, we created an llm instance and called it, passing a string. LangChain itself will send a request to the OpenAI API and return a text response. The result (response) is a string with the generated poem.

For example, the output might be something like this (the model will come up with its own version):

"Far away in the night, stars shine,"
