from config import llm, print_response

def demonstrate_prompt_chaining():
    """
    Demonstrates prompt chaining where output of one prompt becomes input for another:
    1. Translation chain
    2. Analysis chain
    3. Content creation chain
    """
    
    input("\nPress Enter to see Example 1: Translation Chain...")
    # Example 1: Translation Chain
    text_to_translate = "The weather is beautiful today."
    
    input("\nPress Enter to see Step 1: English to French...")
    # Step 1: English to French
    french_prompt = f"""
    Translate this English text to French:
    "{text_to_translate}"
    """
    french_response = llm.invoke(french_prompt)
    french_text = french_response.content
    
    input("\nPress Enter to see Step 2: French to Spanish...")
    
    # Step 2: French to Spanish
    spanish_prompt = f"""
    Translate this French text to Spanish:
    "{french_text}"
    """
    spanish_response = llm.invoke(spanish_prompt)
    spanish_text = spanish_response.content
    
    input("\nPress Enter to see Step 3: Spanish to German...")
    
    # Step 3: Spanish to German
    german_prompt = f"""
    Translate this Spanish text to German:
    "{spanish_text}"
    """
    german_response = llm.invoke(german_prompt)
    
    print_response("Translation Chain", 
                  f"Original: {text_to_translate}\nFrench: {french_text}\nSpanish: {spanish_text}", 
                  german_response)
    
    input("\nPress Enter to see Example 2: Analysis Chain...")
    
    # Example 2: Analysis Chain
    input("\nPress Enter to see Step 1: Generate a business scenario...")
    # Step 1: Generate a business scenario
    scenario_prompt = """
    Generate a brief business scenario about a company facing a challenge.
    Keep it under 100 words.
    """
    scenario_response = llm.invoke(scenario_prompt)
    scenario = scenario_response.content
    
    input("\nPress Enter to see Step 2: Analyze the problems...")
    
    # Step 2: Analyze the problems
    analysis_prompt = f"""
    Identify the key problems in this scenario:
    {scenario}
    
    List each problem separately.
    """
    analysis_response = llm.invoke(analysis_prompt)
    problems = analysis_response.content
    
    input("\nPress Enter to see Step 3: Generate solutions...")
    
    # Step 3: Generate solutions
    solution_prompt = f"""
    Based on these problems:
    {problems}
    
    Provide specific solutions for each problem identified.
    """
    solution_response = llm.invoke(solution_prompt)
    
    print_response("Analysis Chain", 
                  f"Scenario: {scenario}\n\nProblems: {problems}", 
                  solution_response)

    input("\nPress Enter to see Example 3: Content Creation Chain...")
    
    # Example 3: Content Creation Chain
    input("\nPress Enter to see Step 1: Generate article topic...")
    # Step 1: Generate article topic and outline
    topic_prompt = """
    Generate a topic and outline for a technical blog post about an emerging technology trend.
    Include 3-4 main sections.
    """
    topic_response = llm.invoke(topic_prompt)
    topic_outline = topic_response.content
    
    input("\nPress Enter to see Step 2: Expand outline into draft...")
    
    # Step 2: Generate first draft
    draft_prompt = f"""
    Based on this outline:
    {topic_outline}
    
    Write a first draft of the blog post. Keep each section concise but informative.
    """
    draft_response = llm.invoke(draft_prompt)
    draft = draft_response.content
    
    input("\nPress Enter to see Step 3: Polish and finalize...")
    
    # Step 3: Polish and improve
    final_prompt = f"""
    Review and improve this draft blog post:
    {draft}
    
    Make these improvements:
    - Add a compelling introduction
    - Enhance clarity and flow
    - Add a strong conclusion
    - Suggest a catchy title
    """
    final_response = llm.invoke(final_prompt)
    
    print_response("Content Creation Chain",
                  f"Original Outline: {topic_outline}\n\nFirst Draft: {draft}",
                  final_response)

if __name__ == "__main__":
    demonstrate_prompt_chaining() 