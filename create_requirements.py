import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Check if OpenAI API key is available
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# List of required packages
required_packages = [
    "langchain",
    "langchain-community",
    "langchain-openai",
    "openai",
    "python-dotenv",
    "faiss-cpu",
    "chromadb"
]

# Create requirements.txt file
with open("requirements.txt", "w") as f:
    for package in required_packages:
        f.write(f"{package}\n")

print("Created requirements.txt file with the following packages:")
for package in required_packages:
    print(f"- {package}")
