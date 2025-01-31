from typing import List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define our mathematical tools
@tool
def add(a: int, b: int) -> int:
    """Adds two integers a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers a and b."""
    return a * b

def evaluate_tool_calling(llm_with_tools, test_cases: List[dict]):
    """
    Evaluate the LLM's ability to identify when tools should be used.
    
    Args:
        llm_with_tools: LLM with bound tools
        test_cases: List of test cases with queries and expected tool usage
    
    Returns:
        Dictionary containing evaluation metrics
    """
    correct_tool_usage = 0
    total_cases = len(test_cases)
    
    for case in test_cases:
        query = case["query"]
        should_use_tool = case["should_use_tool"]
        
        # Get LLM response
        response = llm_with_tools.invoke(query)
        # earlier, the LLM response was in response.content
        # Check if tool was used (tool_calls will be non-empty if used)
        tool_was_used = len(response.tool_calls) > 0
        
        # Check if the tool usage matches what we expected
        if tool_was_used == should_use_tool:
            correct_tool_usage += 1
            result = "CORRECT"
        else:
            result = "INCORRECT"
            
        print(f"\nQuery: {query}")
        print(f"Expected tool usage: {should_use_tool}")
        print(f"Actual tool usage: {tool_was_used}")
        print(f"Result: {result}")
        if tool_was_used:
            # print(f"Tools called: {[tool.name for tool in response.tool_calls]}")
            print(f"Tools called: {[tool['name'] for tool in response.tool_calls]}")

    accuracy = (correct_tool_usage / total_cases) * 100
    
    return {
        "total_cases": total_cases,
        "correct_predictions": correct_tool_usage,
        "accuracy": accuracy
    }

def main():
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0)
    
    # Create tools list and bind to LLM
    tools = [add, multiply]
    llm_with_tools = llm.bind_tools(tools)
    
    # Define test cases
    test_cases = [
        {"query": "What is 5 plus 3?", "should_use_tool": True},
        {"query": "What is 4 times 7?", "should_use_tool": True},
        {"query": "What is the capital of France?", "should_use_tool": False},
        {"query": "Calculate 10 * 2", "should_use_tool": True},
        {"query": "How's the weather today?", "should_use_tool": False},
        {"query": "What is 15 + 27?", "should_use_tool": True}
    ]
    
    print("Starting tool calling evaluation...")
    print("=" * 50)
    
    # Run evaluation
    results = evaluate_tool_calling(llm_with_tools, test_cases)
    
    # Print final results
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Total test cases: {results['total_cases']}")
    print(f"Correct predictions: {results['correct_predictions']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
