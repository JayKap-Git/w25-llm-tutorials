from config import llm, print_response

def demonstrate_few_shot_prompting():
    """
    Demonstrates different types of few-shot prompting that build upon each other:
    1. Basic few-shot - Simple pattern matching
    2. Classification - Multiple categories and labels
    3. Format transformation - Complex pattern with specific structure
    4. Advanced few-shot - Combining multiple aspects
    """
    
    print("\n=== Few-Shot Prompting Demonstration ===")
    input("Press Enter to see Example 1: Basic Few-Shot prompt...")
    
    # Example 1: Basic Few-Shot (Simple pattern matching)
    basic_prompt = """
    Here are examples of converting numbers to their written form:
    
    Input: 123
    Output: one hundred twenty-three
    
    Input: 45
    Output: forty-five
    
    Input: 7890
    Output: seven thousand eight hundred ninety
    
    Now convert this number:
    Input: 5432
    """
    basic_prompt = """
    Convert the following number to its written form:
    
    Input: 5432
    Output:
    """
    response = llm.invoke(basic_prompt)
    print_response("Basic Few-Shot", basic_prompt, response)
    
    input("\nPress Enter to see Example 2: Classification Few-Shot prompt...")
    
    # Example 2: Classification Few-Shot (Multiple categories)
    classification_prompt = """
    Here are examples of classifying customer feedback:
    
    Feedback: "The product arrived damaged and customer service was unhelpful."
    Sentiment: Negative
    Category: Product Quality, Customer Service
    
    Feedback: "Amazing product! Works exactly as advertised."
    Sentiment: Positive
    Category: Product Quality
    
    Feedback: "Shipping was quick but the instructions were confusing."
    Sentiment: Mixed
    Category: Shipping, Documentation
    
    Now classify this feedback:
    Feedback: "Great features but took forever to arrive and packaging was poor."
    """
    classification_prompt = """
    Classify the following customer feedback:
    
    Feedback: "Great features but took forever to arrive and packaging was poor."

    Your answer should be in the following format:
    Sentiment: [positive, negative, mixed]
    Category: [list of categories in which the feedback falls] 
    """
    response = llm.invoke(classification_prompt)
    print_response("Classification Few-Shot", classification_prompt, response)
    
    input("\nPress Enter to see Example 3: Format Transformation Few-Shot prompt...")
    
    # Example 3: Format Transformation (Complex pattern with structure)
    format_prompt = """
    Here are examples of converting research papers into structured summaries:
    
    Paper: "Effects of Coffee on Productivity in Office Workers"
    Content: A 6-month study conducted across 5 offices with 100 participants examined how coffee consumption affects workplace productivity. Workers were divided into controlled coffee-break groups. Results showed 20% higher task completion rates in groups with scheduled coffee breaks, particularly in afternoon sessions. However, excessive consumption (>6 cups) showed decreased benefits.
    Summary:
    OBJECTIVE: Study coffee's impact on workplace productivity
    METHODS: 6-month controlled study, 100 participants across 5 offices
    FINDINGS: 20% higher task completion with scheduled breaks, diminishing returns after 6 cups
    IMPLICATIONS: Implement structured coffee breaks, monitor consumption levels
    
    Paper: "Machine Learning in Healthcare Diagnostics"
    Content: Meta-analysis covering 50 studies from 2018-2023 evaluated ML algorithms in medical diagnosis. Analysis focused on image-based diagnostics in radiology and pathology. Results showed 15% average improvement in diagnosis accuracy, with 30% faster processing times. Implementation costs and training requirements were identified as major challenges.
    Summary:
    OBJECTIVE: Evaluate ML effectiveness in medical diagnosis
    METHODS: Meta-analysis of 50 studies (2018-2023)
    FINDINGS: 15% accuracy improvement, 30% faster processing
    IMPLICATIONS: Beneficial but consider implementation challenges
    
    Now summarize this paper in the same format:
    Paper: "Impact of Remote Work on Team Collaboration During COVID-19"
    Content: A study of 500 technology companies during 2020-2022 analyzed how forced remote work affected team collaboration. Data collected through surveys, productivity metrics, and communication platform analytics showed a 25% increase in written communication but 15% decrease in spontaneous interactions. Teams reported 30% more documented decisions but 20% longer project completion times. Success factors included structured virtual meetings and async communication tools.
    """
    response = llm.invoke(format_prompt)
    print_response("Format Transformation Few-Shot", format_prompt, response)
    
    input("\nPress Enter to see Example 4: Advanced Few-Shot prompt...")
    
    # Example 4: Advanced Few-Shot (Combining multiple aspects)
    advanced_prompt = """
    Here are examples of analyzing technical problems with solutions and explanations:
    
    Problem: "Website loading slowly during peak hours"
    Context: E-commerce site experiencing 5-second load times between 2-5 PM daily. Traffic increases 3x during these hours. Server monitoring shows 85% CPU usage and 90% memory utilization.
    Analysis:
    - ROOT CAUSE: Resource saturation during high traffic
    - IMPACT: 40% increase in bounce rate, estimated $10K daily revenue loss
    - AFFECTED SYSTEMS: Web servers, database
    Solution: 
    - Implement CDN for static content
    - Add application-level caching
    - Scale database read replicas
    Explanation: CDN reduces server load, caching minimizes database queries, read replicas handle traffic spikes
    
    Problem: "Mobile app crashing on user profile update"
    Context: iOS app version 2.4 crashes for 30% of users when updating profile pictures. Crash logs show memory spikes during image processing. Affects devices with iOS 14+ only.
    Analysis:
    - ROOT CAUSE: Memory leak in image processing pipeline
    - IMPACT: 15% user churn, 500 negative reviews
    - AFFECTED SYSTEMS: iOS app, image processing service
    Solution:
    - Implement progressive image loading
    - Add memory usage monitoring
    - Optimize image processing algorithm
    Explanation: Progressive loading reduces memory spikes, monitoring prevents cascading failures
    
    Now analyze this problem with the same structure:
    Problem: "Database queries timing out during report generation"
    Context: Business intelligence dashboard timing out when generating monthly reports. Affects reports with >100K rows. Database CPU usage hits 100%, queries taking >30 seconds. Problem started after adding three new data metrics last month.
    """
    response = llm.invoke(advanced_prompt)
    print_response("Advanced Few-Shot", advanced_prompt, response)
    
    print("\n=== End of Few-Shot Prompting Demonstration ===")

if __name__ == "__main__":
    demonstrate_few_shot_prompting() 