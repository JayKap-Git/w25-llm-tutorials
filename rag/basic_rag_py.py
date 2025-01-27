# Load environment variables from .env file
from dotenv import load_dotenv
import os
load_dotenv()

print(os.getenv('OPENAI_API_KEY'))
print(os.getenv('LANGSMITH_TRACING'))
print(os.getenv('LANGSMITH_ENDPOINT'))
print(os.getenv('LANGSMITH_API_KEY'))
print(os.getenv('LANGSMITH_PROJECT'))

# Import required langchain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
import yaml
from rouge_score import rouge_scorer

def create_rag_system(file_path):
    """Initialize the RAG system components"""
    
    # Initialize the LLM and embeddings model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Load and split the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    # Print each document split with a separator for readability
    for i, split in enumerate(splits, 1):
        print(f"\n{'='*80}\nDocument Split #{i}\n{'='*80}\n")
        print(split.page_content)
        print()
    # Create and populate the vector store
    vector_store = Chroma(embedding_function=embeddings)
    vector_store.add_documents(documents=splits)
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    return llm, retriever

def get_response_from_llm(llm, context, question):
    """Get response from LLM using context and question"""
    
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you are not able to find the answer in the retrieved context, just say that you don't know. 
    Keep the answer concise and to the point.
    
    Context: {context}"""
    
    # Format the system prompt with context
    system_prompt_filled = system_prompt.format(context=context)
    
    # Create messages for the chat model
    messages = [
        SystemMessage(content=system_prompt_filled),
        HumanMessage(content=question)
    ]
    
    # Get response from LLM
    response = llm.invoke(messages)
    
    return response.content

def answer_question(llm, retriever, question):
    """Answer a question using the RAG system"""
    
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Combine retrieved documents into context
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Get response from LLM
    answer = get_response_from_llm(llm, context, question)
    
    return answer, context

def evaluate_with_rouge(predictions):
    """
    Evaluate RAG system predictions using ROUGE metrics
    
    Args:
        predictions: List of dictionaries containing:
            - question_id: ID of the question
            - category: Question category
            - question: The question text
            - ground_truth: Ground truth answer
            - predicted: Model's predicted answer
            - context: Retrieved context used for answer
    
    Returns:
        tuple: (detailed_results, average_scores)
    """
    # Initialize ROUGE scorer
    # ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics measure the quality of text by comparing it to reference text
    # rouge1: Measures overlap of individual words (unigrams) between the generated and reference texts
    # rouge2: Measures overlap of word pairs (bigrams) - better for capturing fluency
    # rougeL: Measures longest common subsequence - better for capturing sentence structure
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Store results
    results = []
    total_rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    # Calculate ROUGE scores for each prediction
    for pred in predictions:
        # Calculate ROUGE scores
        # Each score is a tuple of (precision, recall, f1)
        # - Precision: What fraction of predicted words were correct
        # - Recall: What fraction of reference words were captured
        # - F1: Harmonic mean of precision and recall (balanced score)
        scores = scorer.score(pred['ground_truth'], pred['predicted'])
        
        # Store results - we use f1 (fmeasure) as it balances precision and recall
        result = {
            **pred,  # Include all original prediction data
            'rouge_scores': {
                'rouge1': scores['rouge1'].fmeasure,  # Unigram F1 score
                'rouge2': scores['rouge2'].fmeasure,  # Bigram F1 score
                'rougeL': scores['rougeL'].fmeasure   # Longest common subsequence F1 score
            }
        }
        results.append(result)
        
        # Update total scores
        total_rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        total_rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        total_rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    
    # Calculate average scores
    num_questions = len(predictions)
    avg_scores = {
        'rouge1': total_rouge_scores['rouge1'] / num_questions,
        'rouge2': total_rouge_scores['rouge2'] / num_questions,
        'rougeL': total_rouge_scores['rougeL'] / num_questions
    }
    
    return results, avg_scores

def print_rouge_evaluation(results, avg_scores):
    """Print ROUGE evaluation results in a formatted way"""
    print("\n=== ROUGE Evaluation Results ===")
    print(f"\nNumber of questions evaluated: {len(results)}")
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.4f}  # Higher score means better word overlap")
    print(f"ROUGE-2: {avg_scores['rouge2']:.4f}  # Higher score means better phrase overlap")
    print(f"ROUGE-L: {avg_scores['rougeL']:.4f}  # Higher score means better sequential overlap")
    
    print("\nDetailed Results:")
    for result in results:
        print(f"\nQuestion {result['question_id']} ({result['category']}):")
        print(f"Q: {result['question']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Predicted: {result['predicted']}")
        print("ROUGE Scores:")
        print(f"  ROUGE-1: {result['rouge_scores']['rouge1']:.4f}")
        print(f"  ROUGE-2: {result['rouge_scores']['rouge2']:.4f}")
        print(f"  ROUGE-L: {result['rouge_scores']['rougeL']:.4f}")
        print(f"  Context: {result['context']}")

def extract_claims(llm, text):
    """Extract atomic claims from a piece of text using the LLM"""
    
    system_prompt = """You are a precise claim extractor. Break down the given text into atomic claims.
    An atomic claim is a simple, declarative statement that cannot be broken down further while maintaining its factual meaning.
    
    Format each claim on a new line starting with a dash (-).
    Only include factual, verifiable claims. Exclude opinions, explanations, and redundant information.
    
    Example:
    Text: "John, who was born in Paris in 1990, moved to London in 2015 for work and got married there in 2018."
    Claims:
    - John was born in Paris
    - John was born in 1990
    - John moved to London in 2015
    - John moved to London for work
    - John got married in London in 2018
    
    Text: {text}
    Claims:"""
    
    messages = [
        SystemMessage(content=system_prompt.format(text=text)),
        HumanMessage(content="Extract the atomic claims from the above text.")
    ]
    
    response = llm.invoke(messages)
    
    # Split the response into individual claims and clean them
    claims = [
        claim.strip('- ').strip() 
        for claim in response.content.split('\n') 
        if claim.strip('- ').strip()
    ]
    
    return claims

def check_claim_entailment(llm, claim, reference_claims):
    """Check if a claim is entailed by any of the reference claims"""
    
    system_prompt = """You are a precise natural language inference system.
    Determine if the given claim is entailed by (logically follows from) any of the reference claims.
    
    A claim is entailed if it is logically necessary given the reference claims.
    The claim must be fully supported by the references, not just partially or possibly true.
    
    Claim: {claim}
    
    Reference Claims:
    {references}
    
    Answer with only 'Yes' if the claim is entailed, or 'No' if it is not entailed.
    Explain your reasoning briefly."""
    
    formatted_refs = "\n".join(f"- {ref}" for ref in reference_claims)
    
    messages = [
        SystemMessage(content=system_prompt.format(
            claim=claim, 
            references=formatted_refs
        )),
        HumanMessage(content="Is the claim entailed by the reference claims?")
    ]
    
    response = llm.invoke(messages)
    
    # Check if response starts with 'Yes'
    return response.content.strip().lower().startswith('yes')

def evaluate_factual_correctness(llm, predictions):
    """
    Evaluate the factual correctness of predictions by comparing claims
    
    Args:
        predictions: List of dictionaries containing ground truth and predicted answers
        
    Returns:
        tuple: (detailed_results, average_score)
    """
    results = []
    total_score = 0
    
    for pred in predictions:
        # Extract claims from both predicted and ground truth answers
        predicted_claims = extract_claims(llm, pred['predicted'])
        reference_claims = extract_claims(llm, pred['ground_truth'])
        
        if not predicted_claims:  # If no claims were extracted
            factual_score = 0
        else:
            # Check each predicted claim against reference claims
            correct_claims = sum(
                1 for claim in predicted_claims 
                if check_claim_entailment(llm, claim, reference_claims)
            )
            factual_score = correct_claims / len(predicted_claims)
        
        result = {
            **pred,  # Include all original prediction data
            'factual_score': factual_score,
            'extracted_claims': {
                'predicted': predicted_claims,
                'reference': reference_claims
            }
        }
        results.append(result)
        total_score += factual_score
    
    avg_score = total_score / len(predictions) if predictions else 0
    return results, avg_score

def print_factual_evaluation(results, avg_score):
    """Print factual correctness evaluation results"""
    print("\n=== Factual Correctness Evaluation Results ===")
    print(f"\nNumber of questions evaluated: {len(results)}")
    print(f"Average Factual Correctness Score: {avg_score:.4f}")
    
    print("\nDetailed Results:")
    for result in results:
        print(f"\nQuestion {result['question_id']} ({result['category']}):")
        print(f"Q: {result['question']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Factual Correctness Score: {result['factual_score']:.4f}")
        print("Extracted Claims:")
        print("  Predicted:")
        for claim in result['extracted_claims']['predicted']:
            print(f"    - {claim}")
        print("  Reference:")
        for claim in result['extracted_claims']['reference']:
            print(f"    - {claim}")

def evaluate_rag_system(llm, retriever):
    """Evaluate the RAG system using predefined QA pairs"""
    
    # Load QA pairs from YAML file
    with open("rag/data/qa_pairs_test.yaml", "r") as file:
        qa_data = yaml.safe_load(file)
    
    # Get predictions for all questions
    predictions = []
    for qa_pair in qa_data['qa_pairs']:
        predicted_answer, context = answer_question(llm, retriever, qa_pair['question'])
        
        prediction = {
            'question_id': qa_pair['id'],
            'category': qa_pair['category'],
            'question': qa_pair['question'],
            'ground_truth': qa_pair['answer'],
            'predicted': predicted_answer,
            'context': context
        }
        predictions.append(prediction)
    
    # Evaluate using ROUGE metrics
    rouge_results, rouge_scores = evaluate_with_rouge(predictions)
    print_rouge_evaluation(rouge_results, rouge_scores)
    
    # Evaluate using Factual Correctness
    factual_results, factual_score = evaluate_factual_correctness(llm, predictions)
    print_factual_evaluation(factual_results, factual_score)
    
    return predictions, {
        'rouge': {
            'results': rouge_results,
            'scores': rouge_scores
        },
        'factual': {
            'results': factual_results,
            'score': factual_score
        }
    }

def main():
    # Initialize the RAG system
    file_path = "rag/data/story_1.txt"
    llm, retriever = create_rag_system(file_path)
    
    # Run evaluation
    predictions, evaluation_results = evaluate_rag_system(llm, retriever)
    
    # Access specific evaluation results if needed
    rouge_results = evaluation_results['rouge']

if __name__ == "__main__":
    main()

