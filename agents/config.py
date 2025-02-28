import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("/Users/jayantkapoor/Documents/GitHub/w25-llm-tutorials/.env")

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Initialize the OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    # model="gpt-o1",
)

def print_response(title, prompt, response):
    """Helper function to print prompts and responses"""
    print(f"\n=== {title} ===")
    print("\nPrompt:")
    print(prompt)
    print("\nResponse:")
    print(response.content)
    print("\n" + "="*50)