from config import llm, print_response

def demonstrate_chain_of_thought():
    """
    Demonstrates different applications of chain-of-thought prompting:
    1. Mathematical reasoning
    2. Logical deduction
    3. Problem decomposition
    """
    input("\nPress Enter to see Example 1: Mathematical Reasoning...")
    
    # Example 1: Mathematical Reasoning
    math_prompt = """
    Let's solve this math problem step by step:
    
    Problem: A store has a 20% off sale. If you buy a shirt originally priced at $80 
    and use a $10 coupon after the sale discount, what is the final price?
    
    Let's think about this step by step.
    
    What's the solution?
    """
    response = llm.invoke(math_prompt)
    print_response("Mathematical Reasoning", math_prompt, response)

    input("\nPress Enter to see Example 2: Logical Deduction...")
    
    # Example 2: Logical Deduction
    logic_prompt = """
    Let's solve this logic puzzle:
    
    Puzzle: If all A are B, and some B are C, what can we conclude about A and C?
    
    Let's reason through this step by step.
    
    What can we conclude?
    """
    response = llm.invoke(logic_prompt)
    print_response("Logical Deduction", logic_prompt, response)
    
    input("\nPress Enter to see Example 3: Problem Decomposition...")
    
    # Example 3: Problem Decomposition
    decomposition_prompt = """
    Let's break down this complex software development task into manageable components:
    
    Task: Build a real-time collaborative document editor with version history
    
    Let's decompose this problem systematically:

    1) First, identify the core components:
       - List all major features required
       - Determine technical domains involved
       - Identify potential dependencies
    
    2) Break down each component:
       a) Document Editor Base
       
       b) Real-time Collaboration
          
       c) Version History
          
       d) Infrastructure
        
    3) Analyze technical challenges:
       - Identify potential bottlenecks
       - List security considerations
       - Note scalability requirements
    
    4) Propose implementation strategy:
       - Suggest technology stack
       - Outline development phases
       - Define milestones
       
    5) Consider maintenance aspects:
       - Monitoring requirements
       - Backup strategies
       - Update procedures
    
    Please provide a detailed breakdown following these steps, including specific technical recommendations and potential pitfalls to avoid.
    """
    response = llm.invoke(decomposition_prompt)
    print_response("Problem Decomposition", decomposition_prompt, response)

if __name__ == "__main__":
    demonstrate_chain_of_thought() 