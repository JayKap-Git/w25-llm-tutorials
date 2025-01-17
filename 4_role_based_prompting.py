from config import llm, print_response

def demonstrate_role_based_prompting():
    """
    Demonstrates different roles and perspectives in prompting:
    1. Expert roles
    2. Character roles
    3. Professional roles
    4. Teaching roles
    """
    input("\nPress Enter to see Example 1: Expert Role...")
    
    # Example 1: Expert Role
    expert_prompt = """
    Role: You are a senior data scientist with 10 years of experience in machine learning.
    
    Task: Explain the trade-offs between using Random Forests vs Neural Networks 
    for a classification problem with structured data.
    
    Consider:
    - Performance
    - Interpretability
    - Training time
    - Data requirements
    """
    response = llm.invoke(expert_prompt)
    print_response("Expert Role", expert_prompt, response)
    
    input("\nPress Enter to see Example 2: Teaching Role...")
    
    # Example 2: Teaching Role
    teacher_prompt = """
    Role: You are a high school physics teacher known for making complex concepts simple.
    
    Task: Explain quantum entanglement to your students.
    
    Requirements:
    - Use everyday analogies
    - Keep it under 5 minutes
    - Include a simple example
    - Address common misconceptions
    """
    response = llm.invoke(teacher_prompt)
    print_response("Teaching Role", teacher_prompt, response)
    
    input("\nPress Enter to see Example 3: Professional Role...")
    
    # Example 3: Professional Role
    professional_prompt = """
    Role: You are a senior marketing manager at a tech company.
    
    Task: Write a brief proposal for launching a new AI-powered smartphone app.
    
    Include:
    - Target audience
    - Unique selling points
    - Marketing channels
    - Success metrics
    """
    response = llm.invoke(professional_prompt)
    print_response("Professional Role", professional_prompt, response)
    
    input("\nPress Enter to see Example 4: Character Role...")
    
    # Example 4: Character Role
    character_prompt = """
    Role: You are a detective in a noir novel.
    
    Task: Investigate a murder case where a wealthy businessman was found dead in his office on the 40th floor of a downtown skyscraper.
    
    Case Details:
    - Victim: James Morrison, 52, CEO of Morrison Technologies
    - Time of Death: Between 9-11pm last night
    - Cause: Blunt force trauma to the head
    
    Consider:
    - Crime Scene: 
      * Broken whiskey glass on the desk
      * Window slightly ajar despite the height
      * Victim's laptop missing
      * Safe left open and empty
    
    - Witnesses:
      * Night security guard who heard a loud noise
      * Cleaning lady who saw lights on at 10pm
      * Secretary who left at 8pm
    
    - Suspects:
      * Business partner with a recent falling out
      * Ex-wife who was cut from the will
      * Young VP gunning for the CEO position
      * Unknown person seen entering building
    
    Analyze the evidence and develop your theory of the crime.
    """
    response = llm.invoke(character_prompt)
    print_response("Character Role", character_prompt, response)
    
if __name__ == "__main__":
    demonstrate_role_based_prompting() 