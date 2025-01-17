from config import llm, print_response

def demonstrate_self_consistency():
    """
    Demonstrates self-consistency prompting where the same prompt is asked multiple times
    and the majority answer is taken as the final result. This helps improve reliability
    by aggregating multiple responses through voting.
    """
    
    input("\nPress Enter to see Example 1: Mathematical Problem...")
    # Example 1: Mathematical Problem
    math_prompt = """
    Solve this math problem step by step:
    
    In a class of 30 students, 60% are girls. If 3 boys leave and 2 girls join,
    what percentage of the class is now girls?
    """
    
    # Get 3 separate responses
    math_responses = []
    for i in range(3):
        response = llm.invoke(math_prompt)
        math_responses.append(response)
    
    print_response("Mathematical Self-Consistency", 
                  f"Original Prompt:\n{math_prompt}\n\nThree Independent Solutions:",
                  "\n\nSolution 1:\n" + math_responses[0] + 
                  "\n\nSolution 2:\n" + math_responses[1] +
                  "\n\nSolution 3:\n" + math_responses[2])
    
    # Example 2: Text Analysis
    analysis_prompt = """
    What is the main message of this quote? Explain your interpretation:
    
    "The only way to do great work is to love what you do." - Steve Jobs
    """
    
    input("\nPress Enter to see Example 2: Text Analysis...")
    # Get 3 separate responses
    analysis_responses = []
    for i in range(3):
        response = llm.invoke(analysis_prompt)
        analysis_responses.append(response)
        
    print_response("Analysis Self-Consistency",
                  f"Original Prompt:\n{analysis_prompt}\n\nThree Independent Interpretations:",
                  "\n\nInterpretation 1:\n" + analysis_responses[0] +
                  "\n\nInterpretation 2:\n" + analysis_responses[1] +
                  "\n\nInterpretation 3:\n" + analysis_responses[2])
    
    input("\nPress Enter to see Example 3: Decision Making...")
    # Example 3: Decision Making
    decision_prompt = """
    Should a company switch to a fully remote work model? Consider the pros and cons
    and make a clear recommendation (Yes or No) with your reasoning.
    """
    
    # Get 3 separate responses
    decision_responses = []
    for i in range(3):
        response = llm.invoke(decision_prompt)
        decision_responses.append(response)
        
    print_response("Decision Self-Consistency",
                  f"Original Prompt:\n{decision_prompt}\n\nThree Independent Recommendations:",
                  "\n\nRecommendation 1:\n" + decision_responses[0] +
                  "\n\nRecommendation 2:\n" + decision_responses[1] +
                  "\n\nRecommendation 3:\n" + decision_responses[2])

if __name__ == "__main__":
    demonstrate_self_consistency()