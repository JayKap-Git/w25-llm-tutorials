from config import llm, print_response

def demonstrate_tree_of_thoughts():
    """
    Demonstrates tree of thoughts prompting where multiple experts explore different reasoning paths:
    1. Complex ethical dilemma analysis
    2. Software architecture design
    """
    
    print("\n=== Tree of Thoughts Demonstration ===")
    input("Press Enter to see Example 1: Ethical Dilemma Analysis...")
    
    # Example 1: Ethical Dilemma Analysis
    ethics_prompt = """
    Imagine three different ethicists are analyzing this complex dilemma.
    Each ethicist will share one key consideration or argument,
    then others will respond and build on it.
    If any ethicist changes their view based on others' arguments, they should explain why.
    
    The ethicists should consider:
    1) Core ethical principles at stake
    2) Stakeholder impacts
    3) Precedent implications
    4) Potential unintended consequences
    5) Alternative approaches
    
    After exploring perspectives, the ethicists should:
    1) Identify areas of agreement/disagreement
    2) Weigh competing principles
    3) Propose actionable recommendations
    
    The dilemma is: A company has developed an AI that can predict mental health crises 
    with 95% accuracy by analyzing social media posts. Should they deploy it publicly?
    Consider privacy, autonomy, beneficence, and justice.
    """
    response = llm.invoke(ethics_prompt)
    print_response("Ethical Dilemma Analysis", ethics_prompt, response)
    
    input("Press Enter to see Example 2: Software Architecture Design...")
    
    # Example 2: Software Architecture Design
    architecture_prompt = """
    Imagine three different senior software architects are designing a system.
    Each architect will propose one architectural decision or component,
    then others will evaluate and build upon it.
    If any architect sees potential issues, they should propose alternatives.
    
    The architects should consider:
    1) Scalability requirements
    2) Security implications
    3) Data flow and storage
    4) Integration points
    5) Failure modes
    
    After exploring designs, the architects should:
    1) Evaluate trade-offs
    2) Identify potential bottlenecks
    3) Present final architecture
    
    The task is: Design a real-time collaborative document editing system 
    that must support 100,000 concurrent users, ensure consistency across devices,
    handle offline editing, and maintain version history for regulatory compliance.
    """
    response = llm.invoke(architecture_prompt)
    print_response("Software Architecture Design", architecture_prompt, response)

if __name__ == "__main__":
    demonstrate_tree_of_thoughts()