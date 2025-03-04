from typing import List, Dict, Any, TypedDict, Annotated, Literal, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import os
from pydantic import BaseModel, Field

# Initialize the LLM with the specified configuration
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "fake-key"

llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    temperature=0,
    api_key="fake-key",
    base_url="http://localhost:1234/v1"
)

# Define event types
class EventType:
    NEW_QUERY = "new_query"
    NEED_RESEARCH = "need_research"
    NEED_ANALYSIS = "need_analysis"
    NEED_WRITING = "need_writing"
    NEED_SUMMARY = "need_summary"
    NEED_CLARIFICATION = "need_clarification"
    COMPLETED = "completed"

# Define the state schema
class Event(BaseModel):
    event_type: str
    content: str = ""

class AgentState(TypedDict):
    messages: List[Any]
    events: List[Event]
    research_data: str
    analysis_data: str
    writing_data: str
    clarification_requests: List[str]
    final_answer: str
    current_event: Event

# Event dispatcher - determines what happens next based on current event
def event_dispatcher(state: AgentState) -> Literal["process_query", "do_research", "do_analysis", "do_writing", "request_clarification", "generate_summary", "END"]:
    """Routes the workflow based on the current event."""
    current_event = state["current_event"]
    
    if current_event.event_type == EventType.NEW_QUERY:
        return "process_query"
    elif current_event.event_type == EventType.NEED_RESEARCH:
        return "do_research"
    elif current_event.event_type == EventType.NEED_ANALYSIS:
        return "do_analysis"
    elif current_event.event_type == EventType.NEED_WRITING:
        return "do_writing"
    elif current_event.event_type == EventType.NEED_CLARIFICATION:
        return "request_clarification"
    elif current_event.event_type == EventType.NEED_SUMMARY:
        return "generate_summary"
    elif current_event.event_type == EventType.COMPLETED:
        return "END"
    else:
        # Default fallback
        return "process_query"

# Query processor - determines what events need to be triggered based on the query
def process_query(state: AgentState) -> AgentState:
    """Analyzes the query and determines what events to trigger."""
    messages = state["messages"]
    
    # Create a prompt to determine what needs to be done
    prompt = SystemMessage(content="""You are an event coordinator in a workflow system.
Analyze this query and determine what events should be triggered.
Possible events are:
- need_research: If information gathering is required
- need_analysis: If analytical processing is needed
- need_writing: If content creation is needed
- need_clarification: If the query is ambiguous or incomplete
- need_summary: If we have enough information to summarize

For a typical informational query, start with research.
For opinion or creative tasks, analysis or writing may be more appropriate.
For unclear queries, request clarification.

Respond with ONLY ONE of these event types - the most appropriate first step.""")
    
    response = llm.invoke([prompt] + messages)
    decision = response.content.strip().lower()
    
    # Determine the next event based on the response
    if "research" in decision:
        next_event = Event(event_type=EventType.NEED_RESEARCH)
    elif "analysis" in decision:
        next_event = Event(event_type=EventType.NEED_ANALYSIS)
    elif "writing" in decision:
        next_event = Event(event_type=EventType.NEED_WRITING)
    elif "clarification" in decision:
        next_event = Event(event_type=EventType.NEED_CLARIFICATION)
    elif "summary" in decision:
        next_event = Event(event_type=EventType.NEED_SUMMARY)
    else:
        # Default to research if unclear
        next_event = Event(event_type=EventType.NEED_RESEARCH)
    
    # Update the state with the new event
    state["events"].append(next_event)
    state["current_event"] = next_event
    
    return state

# Research handler
def do_research(state: AgentState) -> AgentState:
    """Performs research based on the query."""
    messages = state["messages"]
    
    prompt = SystemMessage(content="""You are a research specialist.
Gather relevant information related to the query.
Focus on facts, data points, and background information.
Be comprehensive but concise.""")
    
    response = llm.invoke([prompt] + messages)
    research_data = response.content
    
    # Store the research results
    state["research_data"] = research_data
    
    # Determine next event
    analysis_prompt = SystemMessage(content="""Based on the research that has been done,
should we proceed to analysis, writing, or request clarification?
Respond with ONLY ONE of: need_analysis, need_writing, need_clarification""")
    
    next_decision = llm.invoke([analysis_prompt] + messages + [
        AIMessage(content=f"Research findings: {research_data}")
    ])
    
    decision = next_decision.content.strip().lower() or "writing"

    
    if "analysis" in decision:
        next_event = Event(event_type=EventType.NEED_ANALYSIS)
    elif "writing" in decision:
        next_event = Event(event_type=EventType.NEED_WRITING)
    elif "clarification" in decision:
        next_event = Event(event_type=EventType.NEED_CLARIFICATION)
    else:
        # Default to analysis
        next_event = Event(event_type=EventType.NEED_ANALYSIS)
    
    state["events"].append(next_event)
    state["current_event"] = next_event
    
    return state    

# Analysis handler
def do_analysis(state: AgentState) -> AgentState:
    """Performs analysis based on research data."""
    messages = state["messages"]
    research_data = state["research_data"]
    
    prompt = SystemMessage(content=f"""You are an analysis specialist.
Based on the following research information:
{research_data}

Analyze this information in relation to the query.
Identify patterns, implications, and insights.
Draw meaningful conclusions.""")
    
    response = llm.invoke([prompt] + messages)
    analysis_data = response.content
    
    # Store the analysis results
    state["analysis_data"] = analysis_data
    
    # Determine next event
    next_prompt = SystemMessage(content="""Based on the analysis that has been done,
should we proceed to writing, request clarification, or finalize with a summary?
Respond with ONLY ONE of: need_writing, need_clarification, need_summary""")
    
    next_decision = llm.invoke([next_prompt] + messages + [
        AIMessage(content=f"Analysis: {analysis_data}")
    ])
    
    decision = next_decision.content.strip().lower()
    print(f"Analysis Decision: {decision}")  # Debugging output

    next_event = Event(event_type=EventType.NEED_WRITING)  # Force next step
    state["events"].append(next_event)
    state["current_event"] = next_event

    
    if "writing" in decision:
        next_event = Event(event_type=EventType.NEED_WRITING)
    elif "clarification" in decision:
        next_event = Event(event_type=EventType.NEED_CLARIFICATION)
    elif "summary" in decision:
        next_event = Event(event_type=EventType.NEED_SUMMARY)
    else:
        # Default to writing
        next_event = Event(event_type=EventType.NEED_WRITING)
    
    state["events"].append(next_event)
    state["current_event"] = next_event
    
    return state

# Writing handler
def do_writing(state: AgentState) -> AgentState:
    """Creates polished content based on research and analysis."""
    messages = state["messages"]
    research_data = state["research_data"]
    analysis_data = state["analysis_data"]
    
    prompt = SystemMessage(content=f"""You are a content creation specialist.
Based on the following information:
Research: {research_data}
Analysis: {analysis_data}

Create well-structured, polished content that addresses the query.
Focus on clarity, cohesion, and engaging presentation.""")
    
    response = llm.invoke([prompt] + messages)
    writing_data = response.content
    
    # Store the writing results
    state["writing_data"] = writing_data
    
    # Move to summary
    next_event = Event(event_type=EventType.NEED_SUMMARY)
    state["events"].append(next_event)
    state["current_event"] = next_event
    
    return state

# Clarification handler
def request_clarification(state: AgentState) -> AgentState:
    """Generates a clarification request for ambiguous queries."""
    messages = state["messages"]
    
    prompt = SystemMessage(content="""The query requires clarification.
Identify what specific information is missing or unclear.
Formulate a precise question to get the needed information.""")
    
    response = llm.invoke([prompt] + messages)
    clarification = response.content
    
    # Store the clarification request
    state["clarification_requests"].append(clarification)
    
    # In a real system, we would wait for user input here
    # For demonstration, we'll assume clarification was received and move to research
    
    # Since we can't actually get user input in this example,
    # we'll just move to the next most logical step
    next_event = Event(event_type=EventType.NEED_RESEARCH)
    state["events"].append(next_event)
    state["current_event"] = next_event
    
    return state

# Summary generator
def generate_summary(state: AgentState) -> AgentState:
    """Generates the final summary based on all available information."""
    messages = state["messages"]
    research_data = state["research_data"]
    analysis_data = state["analysis_data"]
    writing_data = state["writing_data"]
    
    prompt = SystemMessage(content=f"""Create a comprehensive final response based on:
Research: {research_data}
Analysis: {analysis_data}
Written Content: {writing_data}

Integrate all this information into a cohesive, complete answer to the original query.""")
    
    response = llm.invoke([prompt] + messages)
    final_answer = response.content
    
    # Store the final answer
    state["final_answer"] = final_answer
    
    # Mark as completed
    next_event = Event(event_type=EventType.COMPLETED)
    state["events"].append(next_event)
    state["current_event"] = next_event
    
    return state

# Build the event-driven workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("process_query", process_query)
workflow.add_node("do_research", do_research)
workflow.add_node("do_analysis", do_analysis)
workflow.add_node("do_writing", do_writing)
workflow.add_node("request_clarification", request_clarification)
workflow.add_node("generate_summary", generate_summary)

# Add conditional edges based on events
workflow.add_conditional_edges(
    "process_query",
    event_dispatcher,
    {
        "process_query": "process_query",
        "do_research": "do_research",
        "do_analysis": "do_analysis",
        "do_writing": "do_writing",
        "request_clarification": "request_clarification",
        "generate_summary": "generate_summary",
        "END": END
    }
)

# Set the entry point
workflow.set_entry_point("process_query")

# Compile the graph
event_driven_agent = workflow.compile()

# Function to run the event-driven workflow
def run_event_driven_workflow(query: str) -> str:
    """Run the event-driven workflow with a user query."""
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "events": [Event(event_type=EventType.NEW_QUERY)],
        "research_data": "",
        "analysis_data": "",
        "writing_data": "",
        "clarification_requests": [],
        "final_answer": "",
        "current_event": Event(event_type=EventType.NEW_QUERY)
    }
    
    # Execute the graph
    result = event_driven_agent.invoke(initial_state)
    
    return result 

# Example usage
if __name__ == "__main__":
    user_query = "What are the environmental impacts of electric vehicles compared to traditional combustion engines?"
    result = run_event_driven_workflow(user_query)  # Now returns full state
    
    print(result["final_answer"])  # Print final answer
    
    print("\nEvent trace:")
    for event in result["events"]:  # Now result is properly defined
        print(f"- {event.event_type}")
