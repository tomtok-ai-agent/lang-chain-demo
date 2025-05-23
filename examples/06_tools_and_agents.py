"""
Example 6: Using Tools and Agents for External API Access

This example demonstrates how to create a simple tool that can be used by an agent
to access external data - in this case, a simulated weather API.

What it solves:
- Allows LLMs to access external data and perform actions
- Agent dynamically decides when to use tools
- Enables more complex interactions beyond text generation
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent

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
    
    # Return weather if city is in our data, otherwise return a default message
    return f"Current weather in {city}: {weather_data.get(city, 'Data not available, but likely pleasant')}"

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Wrap the weather function in a Tool for the agent
    weather_tool = Tool(
        name="WeatherAPI",
        func=lambda q: get_weather(q),
        description="Provides current weather in the specified city. Input should be a city name in English."
    )
    
    # Create LLM for the agent (using gpt-3.5-turbo)
    agent_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Initialize the agent with our tool
    # Using ZeroShotAgent (ReAct) with tool descriptions
    # Added handle_parsing_errors=True to handle output parsing errors
    agent = initialize_agent(
        [weather_tool], 
        agent_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        handle_parsing_errors=True  # Add this parameter to handle parsing errors
    )
    
    # Try asking the agent questions
    queries = [
        "What's the weather like in Tokyo today?",
        "I'm planning a trip to Paris. What's the weather there?",
        "Tell me a joke about the weather in London."
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        try:
            response = agent.invoke({"input": query})
            print(f"Final answer: {response['output']}")
        except Exception as e:
            print(f"Error processing query: {e}")
            print("The agent might need more specific instructions for this type of query.")

if __name__ == "__main__":
    main()
