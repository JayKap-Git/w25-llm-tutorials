from typing import Dict, Any
import json
from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config import llm, print_response

# Simple tool implementations
def add_numbers(a: float, b: float):
    return a - b

def multiply_numbers(a: float, b: float):
    return a * b

# Tool definitions that will be part of the system prompt
TOOL_DEFINITIONS = """
You have access to the following tools:

1. add_numbers:
   - Description: Adds two numbers together
   - Parameters: 
     - a (number): First number
     - b (number): Second number
   - Returns: Sum of the two numbers

2. multiply_numbers:
   - Description: Multiplies two numbers together
   - Parameters:
     - a (number): First number
     - b (number): Second number
   - Returns: Product of the two numbers

Important: Only use these tools when explicitly needed for mathematical calculations.
If the user's query doesn't require mathematical operations, respond directly without using any tools.

You must respond in the following JSON format:
{
    "thoughts": "Your step-by-step reasoning",
    "tool_needed": true/false,
    "tool_name": "name of the tool to use (if tool_needed is true)",
    "tool_args": {"a": number, "b": number} (if tool_needed is true),
    "final_answer": "Your final response to the user"
}

When a tool is needed, you must not give the final answer directly (without using the tool).
"""

def create_chat_completion(
    client: OpenAI,
    user_query: str,
    system_prompt: str = TOOL_DEFINITIONS,
):
    """
    Create a chat completion, process any tool calls, and generate final response.
    """
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0,
    #     # response_format={"type": "json_object"}
    # )
    
    # First LLM call to determine if tool is needed
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    print("Initial LLM Response:", response.content)
    parsed_response = json.loads(response.content)
    
    # If tool is needed, execute it and get final response
    if parsed_response["tool_needed"]:
        tool_result = process_tool_call(
            parsed_response["tool_name"],
            parsed_response["tool_args"]
        )
        print(f"Tool Execution Result: {tool_result}")
        
        # Second LLM call to process tool result
        result_prompt = f"""
Based on the tool execution result of {tool_result}, please provide a final response to the user's query: "{user_query}"

Respond in the following JSON format:
{{
    "thoughts": "Share your thoughts here. You should use the tool result to give a final answer. Do not use your own reasoning. Tool result is absolutely correct.",
    "final_answer": "Your final response incorporating the tool result"
}}
"""
        result_messages = [
            HumanMessage(content=result_prompt)
        ]
        
        final_response = llm.invoke(result_messages)
        print("Final LLM Response:", final_response.content)
        final_parsed = json.loads(final_response.content)
        
        # Update the original response with the final answer
        parsed_response["final_answer"] = final_parsed["final_answer"]
        parsed_response["thoughts"] += f"\nAfter tool execution: {final_parsed['thoughts']}"
    
    return parsed_response

def process_tool_call(tool_name: str, tool_args: Dict[str, float]) -> float:
    """
    Execute the specified tool with given arguments.
    """
    tools = {
        "add_numbers": add_numbers,
        "multiply_numbers": multiply_numbers,
    }
    
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    return tools[tool_name](**tool_args)

def main():
    client = OpenAI()
    
    # Example queries to test the system
    test_queries = [
        "What is 15 plus 27?",
        "What is 15121311 plus 2712124?",
        "Can you multiply 8 by 6?",
        "Can you multiply 716162 by 18182?",
        "What's the weather like today?",
        "Tell me a joke.",
    ]
    
    for query in test_queries:
        print(f"\nUser Query: {query}")
        print("-" * 50)
        
        # Get LLM's response (including tool processing if needed)
        response = create_chat_completion(client, query)
        
        print(f"Final Answer: {response['final_answer']}")
        print(f"Thoughts: {response['thoughts']}")

if __name__ == "__main__":
    main()