"""
Example 9: Combining Agents with Memory and Tools

This example demonstrates a more complex setup combining an agent with memory and tools,
creating a conversational assistant that can remember context and use external tools.

What it solves:
- Creates a more powerful assistant that can both remember conversation history
- Allows the assistant to use tools when needed to answer questions
- Demonstrates how to combine multiple LangChain components
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

def get_weather(city: str) -> str:
    """Simulated function to get weather data for a city"""
    # In a real application, this would call an actual weather API
    weather_data = {
        "New York": "Sunny, 75°F (24°C)",
        "London": "Rainy, 62°F (16°C)",
        "Tokyo": "Cloudy, 70°F (21°C)",
        "Paris": "Partly cloudy, 68°F (20°C)",
        "Sydney": "Clear, 80°F (27°C)"
    }
    
    return f"Current weather in {city}: {weather_data.get(city, 'Data not available, but likely pleasant')}"

def get_capital(country: str) -> str:
    """Simulated function to get the capital of a country"""
    capitals = {
        "USA": "Washington, D.C.",
        "UK": "London",
        "France": "Paris",
        "Germany": "Berlin",
        "Japan": "Tokyo",
        "Australia": "Canberra",
        "Brazil": "Brasília",
        "Canada": "Ottawa",
        "China": "Beijing",
        "India": "New Delhi"
    }
    
    return f"The capital of {country} is {capitals.get(country, 'unknown to this tool')}"

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Create tools for the agent
    tools = [
        Tool(
            name="WeatherAPI",
            func=lambda q: get_weather(q),
            description="Provides current weather in the specified city. Input should be a city name in English."
        ),
        Tool(
            name="CapitalAPI",
            func=lambda q: get_capital(q),
            description="Provides the capital city of the specified country. Input should be a country name in English."
        )
    ]
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create LLM for the agent
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Initialize the conversational agent with memory and tools
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    # Simulate a conversation
    print("Starting conversation with the agent...\n")
    
    queries = [
        "What's the weather like in Tokyo?",
        "What's the capital of France?",
        "And what's the weather there?",  # This should use memory to understand "there" refers to Paris
        "Thank you for the information!"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nUser Query {i}: {query}")
        response = agent.invoke({"input": query})
        print(f"Agent: {response['output']}")

if __name__ == "__main__":
    main()
