from typing import List, Dict, Any, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import os

# Initialize the LLM with the specified configuration
# Setting the base_url directly for OpenAI-compatible API endpoint
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "fake-key"

llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    temperature=0,
    api_key="fake-key",
    base_url="http://localhost:1234/v1"  # Use base_url instead of open_api_base
)

# Define the state schema
class AgentState(TypedDict):
    messages: List[Any]
    next_worker: str
    worker_results: Dict[str, str]
    final_answer: str

# Define the master agent that coordinates and delegates tasks
def master_agent(state: AgentState) -> AgentState:
    """Master agent that decides which worker to delegate to or if processing is complete."""
    
    # Prepare messages for the master agent
    messages = state["messages"]
    worker_results = state["worker_results"]
    
    # Include worker results in the context if available
    results_str = ""
    if worker_results:
        results_str = "\n".join([f"{worker}: {result}" for worker, result in worker_results.items()])
        context_message = f"Worker results so far:\n{results_str}"
        messages = messages + [SystemMessage(content=context_message)]
    
    # Master prompt to analyze the task and determine next steps
    master_prompt = SystemMessage(content="""You are the master coordinator in a hierarchical workflow.
Your job is to:
1. Analyze the user's request
2. Decide which specialized worker to delegate to next (options: 'research_worker', 'analysis_worker', 'writing_worker')
3. If all necessary work is complete, set 'next_worker' to 'END'

Workers have these specialties:
- research_worker: Gathering information and facts
- analysis_worker: Analyzing data and drawing conclusions
- writing_worker: Creating polished final content

Choose the most appropriate worker based on the current needs.""")
    
    response = llm.invoke([master_prompt] + messages)
    
    # Extract the master's decision from the response
    decision = response.content
    
    # Determine which worker to use next or if we're done
    if "research_worker" in decision.lower():
        next_worker = "research_worker"
    elif "analysis_worker" in decision.lower():
        next_worker = "analysis_worker"
    elif "writing_worker" in decision.lower():
        next_worker = "writing_worker"
    elif "end" in decision.lower() or "complete" in decision.lower():
        next_worker = "END"
        # Generate final answer if complete
        final_prompt = SystemMessage(content="""Based on all the work done by the specialized workers, 
        provide a comprehensive final response to the user's request.""")
        final_response = llm.invoke([final_prompt] + messages + [
            SystemMessage(content=f"Worker results:\n{results_str}")
        ])
        state["final_answer"] = final_response.content
    else:
        # Default to research if unclear
        next_worker = "research_worker"
    
    state["next_worker"] = next_worker
    return state

# Define worker agents for specific tasks
def research_worker(state: AgentState) -> AgentState:
    """Worker specialized in research and information gathering."""
    messages = state["messages"]
    
    research_prompt = SystemMessage(content="""You are a research specialist.
Your task is to gather relevant information and facts related to the user's request.
Focus on finding key data points, background information, and important context.
Be thorough but concise in your research.""")
    
    response = llm.invoke([research_prompt] + messages)
    
    # Store the research results
    state["worker_results"]["research_worker"] = response.content
    return state

def analysis_worker(state: AgentState) -> AgentState:
    """Worker specialized in analysis and drawing conclusions."""
    messages = state["messages"]
    worker_results = state["worker_results"]
    
    # Include research results if available
    research_context = ""
    if "research_worker" in worker_results:
        research_context = f"Research information:\n{worker_results['research_worker']}"
    
    analysis_prompt = SystemMessage(content=f"""You are an analysis specialist.
Your task is to analyze the information and draw meaningful conclusions.
{research_context}
Focus on patterns, insights, and implications of the data.
Provide a clear analytical perspective on the user's request.""")
    
    response = llm.invoke([analysis_prompt] + messages)
    
    # Store the analysis results
    state["worker_results"]["analysis_worker"] = response.content
    return state

def writing_worker(state: AgentState) -> AgentState:
    """Worker specialized in creating polished final content."""
    messages = state["messages"]
    worker_results = state["worker_results"]
    
    # Include research and analysis results if available
    context = []
    if "research_worker" in worker_results:
        context.append(f"Research information:\n{worker_results['research_worker']}")
    if "analysis_worker" in worker_results:
        context.append(f"Analysis:\n{worker_results['analysis_worker']}")
    
    context_str = "\n\n".join(context)
    
    writing_prompt = SystemMessage(content=f"""You are a writing specialist.
Your task is to create polished, well-structured content based on the information provided.
{context_str}
Focus on clarity, coherence, and engaging presentation.
Create a comprehensive response that addresses the user's request.""")
    
    response = llm.invoke([writing_prompt] + messages)
    
    # Store the writing results
    state["worker_results"]["writing_worker"] = response.content
    return state

# Define router to determine next steps based on master's decision
def router(state: AgentState) -> str:
    """Route to the next worker or end the workflow."""
    return state["next_worker"]

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("master", master_agent)
workflow.add_node("research_worker", research_worker)
workflow.add_node("analysis_worker", analysis_worker)
workflow.add_node("writing_worker", writing_worker)

# Set the entry point
workflow.set_entry_point("master")

# Add conditional edges from master node
workflow.add_conditional_edges(
    "master",
    router,
    {
        "research_worker": "research_worker",
        "analysis_worker": "analysis_worker",
        "writing_worker": "writing_worker",
        "END": END
    }
)

# Add edges from workers back to master
workflow.add_edge("research_worker", "master")
workflow.add_edge("analysis_worker", "master")
workflow.add_edge("writing_worker", "master")

# Compile the graph
hierarchical_agent = workflow.compile()

# Function to run the hierarchical agent
def run_hierarchical_agent(query: str) -> str:
    """Run the hierarchical agent workflow with a user query."""
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_worker": "",
        "worker_results": {},
        "final_answer": ""
    }
    
    # Execute the graph
    result = hierarchical_agent.invoke(initial_state)
    
    # Return the final answer
    return result["final_answer"]

# Example usage
if __name__ == "__main__":
    user_query = "Explain the impact of artificial intelligence on modern healthcare systems."
    response = run_hierarchical_agent(user_query)
    print(response)