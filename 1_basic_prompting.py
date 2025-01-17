from config import llm, print_response

def demonstrate_basic_prompting():
    """
    Demonstrates different types of basic prompts that build upon each other:
    1. Simple question - Direct question without context
    2. Descriptive instruction - Same question with formatting instructions
    3. Structured format prompt - Adds specific format and organization
    4. Context-providing prompt - Full context with role and purpose
    """
    
    print("\n=== Basic Prompting Demonstration ===")
    input("Press Enter to see Example 1: Simple Question (Basic) prompt...")
    
    # Example 1: Simple Question (Basic)
    simple_prompt = "What is machine learning?"
    response = llm.invoke(simple_prompt)
    print_response("Simple Question (Basic)", simple_prompt, response)
    
    input("\nPress Enter to see Example 2: Descriptive Instruction prompt...")
    
    # Example 2: Descriptive Instruction (Adds clarity and audience)
    descriptive_prompt = """
    Explain machine learning in simple terms that a beginner can understand.
    Use everyday examples to illustrate the concept.
    """
    response = llm.invoke(descriptive_prompt)
    print_response("Descriptive Instruction (With Clarity)", descriptive_prompt, response)
    
    input("\nPress Enter to see Example 3: Structured Format prompt...")
    
    # Example 3: Structured Format (Adds organization and specific sections)
    structured_prompt = """
    Provide a structured explanation of machine learning with these specific sections:

    Section 1: Core Definition
    - Give a one-sentence definition of machine learning

    Section 2: Key Components
    - List the main components of a machine learning system
    - Briefly explain each component's purpose

    Section 3: Basic Process
    - Outline the steps from data to trained model
    - Highlight key decision points

    Format each section clearly with headings and bullet points.
    
    Audience: Beginners in the field of machine learning
    """
    response = llm.invoke(structured_prompt)
    print_response("Structured Format (With Organization)", structured_prompt, response)
    
    input("\nPress Enter to see Example 4: Context-providing prompt...")
    
    # Example 4: Context-providing Prompt (Adds role, purpose, and technical depth)
    context_prompt = """
    Role: You are a senior ML engineer explaining machine learning to new team members.
    
    Context: This explanation will be used in the first week of their training program.
    
    Task: Explain machine learning, covering:
    1. Technical definition and core principles
    2. The ML development lifecycle
    3. Common challenges and best practices
    4. How it fits into the broader AI ecosystem
    
    Requirements:
    - Use technical terminology appropriately
    - Include practical insights from industry experience
    - Address common misconceptions
    - Provide guidance for further learning
    """
    response = llm.invoke(context_prompt)
    print_response("Context-providing Prompt (Complete Framework)", context_prompt, response)
    
    print("\n=== End of Basic Prompting Demonstration ===")

if __name__ == "__main__":
    demonstrate_basic_prompting() 