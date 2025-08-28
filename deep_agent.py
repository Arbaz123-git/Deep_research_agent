# Initialize LLM and Tools

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import time
from groq import RateLimitError

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

validation_llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct",
    temperature=0
)

# Initialize Search Tool
search_tool = TavilySearchResults(max_results=3)

# 3. Define Agent State

from typing import TypedDict, List, Annotated
import operator

class ResearchState(TypedDict):
    question: str
    search_results: List[str]
    drafted_content: str
    refined_answer: str
    follow_up_questions: List[str]
    research_depth: int
    max_depth: int
    validation_score: float  # Optional for quality assessment
    validation_feedback: str  # Optional for quality assessment
    needs_more_research: bool
    
# Helper function for LLM requests with retry
def make_llm_request_with_retry(prompt, max_retries=3, specific_llm=None):
    # Use the provided LLM or default to the main LLM
    target_llm = specific_llm if specific_llm is not None else llm

    for attempt in range(max_retries):
        try:
            response = target_llm.invoke(prompt)
            return response
        except Exception as e:
            if "rate_limit" in str(e).lower() or "413" in str(e) or attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"API limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                raise e
            
# Helper function to summarize content
def summarize_content(content: str, max_length: int = 500) -> str:
    if len(content) <= max_length:
        return content
    
    prompt = f"""Summarize this content in {max_length} characters or less:
    
    {content}
    
    Summary:"""
    
    try:
        response = make_llm_request_with_retry(prompt)
        return response.content[:max_length]
    except:
        return content[:max_length] + "..."
    

# 4. Create Graph Nodes

from langgraph.graph import StateGraph, END

# Define graph builder
builder = StateGraph(ResearchState)

# Node 1: Research - Search the web
def research_node(state: ResearchState):
    results = search_tool.invoke({"query": state["question"]})
    summarized_results = [summarize_content(result["content"]) for result in results]
    return {"search_results": summarized_results}

# Node 2: Draft - Initial content generation
def draft_node(state: ResearchState):
    context = "\n\n".join(state["search_results"])
    
    # Truncate context if too long
    if len(context) > 3000:
        context = context[:3000] + "..."
    
    prompt = f"""Research and answer: {state['question']}
    
    Context:
    {context}
    
    Provide a comprehensive but concise answer:"""
    
    response = make_llm_request_with_retry(prompt)
    return {"drafted_content": response.content}

# Node 3: Refine - Improve the answer
def refine_node(state: ResearchState):
    # Truncate content if too long
    content = state["drafted_content"]
    if len(content) > 2000:
        content = content[:2000] + "..."
    
    prompt = f"""Refine this answer: {content}
    
    Ensure it's:
    1. Well-structured
    2. Contains citations
    3. Addresses all aspects of: {state['question']}
    
    Improved version:"""
    
    response = make_llm_request_with_retry(prompt)
    return {"refined_answer": response.content}

def generate_follow_ups_node(state: ResearchState):
    if state["research_depth"] >= state["max_depth"]:
        return {"follow_up_questions": []}
    
    # Truncate content if too long
    content = state["drafted_content"]
    if len(content) > 1500:
        content = content[:1500] + "..."
    
    prompt = f"""Based on the current research findings, generate 2-3 specific follow-up questions 
    that would help provide a more comprehensive answer to: {state['question']}
    
    Current research:
    {content}
    
    Generate specific, answerable follow-up questions:"""
    
    response = make_llm_request_with_retry(prompt)
    
    # Parse the response to extract questions
    questions = []
    for line in response.content.split('\n'):
        if line.strip() and any(char.isdigit() or char in ['-', '*'] for char in line[:3]):
            questions.append(line.split('.', 1)[-1].strip())
    
    return {"follow_up_questions": questions[:2]}  # Limit to 2 questions

def research_follow_ups_node(state: ResearchState):
    if not state["follow_up_questions"] or state["research_depth"] >= state["max_depth"]:
        return {"search_results": state["search_results"]}
    
    all_results = state["search_results"].copy()
    
    for question in state["follow_up_questions"]:
        results = search_tool.invoke({"query": question})
        for result in results:
            all_results.append(summarize_content(result["content"]))
    
    return {
        "search_results": all_results,
        "research_depth": state["research_depth"] + 1
    }


# New validation node
def validation_node(state: ResearchState):
    # Skip validation if we've already done it
    if state.get("validation_score", 0) > 0.6:  # Good enough score
        return {"needs_more_research": False}
    
    prompt = f"""Evaluate the quality of this research answer for the question: {state['question']}
    
    Answer to evaluate:
    {state['refined_answer']}
    
    Please evaluate based on these criteria:
    1. Completeness (does it address all aspects of the question?)
    2. Accuracy (is the information factually correct?)
    3. Depth (does it provide sufficient detail?)
    4. Structure (is it well-organized?)
    5. Citation quality (are sources properly referenced?)
    
    Provide:
    - A score from 0.0 to 1.0
    - Specific feedback on what could be improved
    - A boolean indicating if more research is needed
    
    Format your response as:
    Score: [0.0-1.0]
    Feedback: [your feedback]
    More Research Needed: [true/false]"""
    try:
        response = make_llm_request_with_retry(prompt, specific_llm=validation_llm)
        content = response.content

        # Parse the response
        score = 0.5  # Default
        feedback = "No specific feedback provided"
        needs_more_research = False
        
        # Extract score
        if "Score:" in content:
            try:
                score_str = content.split("Score:")[1].split("\n")[0].strip()
                score = float(score_str)
            except:
                pass

        # Extract feedback
        if "Feedback:" in content:
            feedback = content.split("Feedback:")[1].split("More Research Needed:")[0].strip()

        # Extract research recommendation
        if "More Research Needed:" in content:
            research_str = content.split("More Research Needed:")[1].strip().lower()
            needs_more_research = "true" in research_str or "yes" in research_str
        
        return {
            "validation_score": score,
            "validation_feedback": feedback,
            "needs_more_research": needs_more_research and state["research_depth"] < state["max_depth"]
        }
    except Exception as e:
        print(f"Validation error: {e}")
        return {
            "validation_score": 0.5,
            "validation_feedback": f"Validation failed: {str(e)}",
            "needs_more_research": False
        }
    
# New research improvement node
def improve_research_node(state: ResearchState):
    if not state["needs_more_research"]:
        return {"follow_up_questions": []}
    
    prompt = f"""Based on this validation feedback, generate specific follow-up research questions:
    
    Original question: {state['question']}
    Current answer: {state['refined_answer']}
    Validation feedback: {state['validation_feedback']}
    
    Generate 2-3 specific research questions that would address the validation feedback:"""

    response = make_llm_request_with_retry(prompt)

    # Parse the response to extract questions
    questions = []
    for line in response.content.split('\n'):
        if line.strip() and any(char.isdigit() or char in ['-', '*'] for char in line[:3]):
            questions.append(line.split('.', 1)[-1].strip())
    
    return {"follow_up_questions": questions[:2]}  # Limit to 2 questions
        

# Build Graph
def create_research_graph():
    builder = StateGraph(ResearchState)

    # Add nodes to graph
    builder.add_node("research", research_node)
    builder.add_node("draft", draft_node)
    builder.add_node("refine", refine_node)
    builder.add_node("generate_follow_ups", generate_follow_ups_node)
    builder.add_node("research_follow_ups", research_follow_ups_node)
    builder.add_node("validation", validation_node)
    builder.add_node("improve_research", improve_research_node)


    # 5. Define Graph Structure

    # Set entry point
    builder.set_entry_point("research")

    # Create edges
    builder.add_edge("research", "draft")
    builder.add_edge("draft", "generate_follow_ups")

    def should_continue_research(state: ResearchState):
        if state["follow_up_questions"] and state["research_depth"] < state["max_depth"]:
            return "research_follow_ups"
        return "refine"

    builder.add_conditional_edges(
        "generate_follow_ups",
        should_continue_research,
        {
            "research_follow_ups": "research_follow_ups",
            "refine": "refine"
        }
    )

    builder.add_edge("research_follow_ups", "draft")
    builder.add_edge("refine", "validation")

    # Conditional edge for validation
    def should_improve_research(state: ResearchState):
        if state["needs_more_research"]:
            return "improve_research"
        return END

    builder.add_conditional_edges(
        "validation",
        should_improve_research,
        {
            "improve_research": "improve_research",
            END: END
        }
    )

    builder.add_edge("improve_research", "research_follow_ups")

    # Compile graph
    return builder.compile()

# Create the research graph
research_graph = create_research_graph()

def conduct_research(question, max_depth=2):
    """
    Conduct deep research on a given question using the research agent.
    
    Args:
        question (str): The research question to investigate
        max_depth (int, optional): Maximum research iterations. Defaults to 2.
    
    Returns:
        dict: Research results containing the refined answer, validation score, and feedback
    """
    print(f"Starting research on: {question}")
    print(f"Maximum research depth: {max_depth}")
    
    # Run the graph
    final_state = research_graph.invoke({
        "question": question,
        "search_results": [],
        "drafted_content": "",
        "refined_answer": "",
        "follow_up_questions": [],
        "research_depth": 0,
        "max_depth": max_depth,
        "validation_score": 0.0,
        "validation_feedback": "",
        "needs_more_research": False
    })
    
    return final_state

# Run the graph
if __name__ == "__main__":
    # Example usage
    question = input("Enter your research question: ")
    max_depth = int(input("Enter maximum research depth (1-3): ") or "2")
    
    results = conduct_research(question, max_depth)
    
    print("\n" + "="*80)
    print("RESEARCH COMPLETE")
    print("="*80)
    print("\nFinal refined answer:")
    print(results["refined_answer"])
    print("\nValidation Score:", results["validation_score"])
    print("Validation Feedback:", results["validation_feedback"])
    
    # Save results to a file
    with open("research_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Research Question: {question}\n")
        f.write(f"Validation Score: {results['validation_score']}\n")
        f.write(f"Validation Feedback: {results['validation_feedback']}\n")
        f.write("\nResearch Results:\n")
        f.write(results["refined_answer"])
    
    print("\nResults saved to research_results.txt")
    