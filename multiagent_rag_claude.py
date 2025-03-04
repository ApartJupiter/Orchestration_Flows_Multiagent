import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Dependencies you'll need to install
# pip install langgraph langchain langchain-openai langchain-community faiss-cpu

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# Initialize LLM with your local server configuration
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # Your local server endpoint
    model="deepseek-r1-distill-qwen-7b",  # Your local model
    temperature=0,
    api_key="fake-key"  # The fake key you're using
)

# For embeddings, we'll use OpenAI's embeddings by default
# If you have a local embeddings server, you might want to replace this
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:1234/v1",
    api_key="fake-key"
)

# Define state schema
class AgentState(BaseModel):
    query: str = Field(description="The original user query")
    context: List[str] = Field(default_factory=list, description="Retrieved documents for context")
    refined_query: Optional[str] = Field(default=None, description="Query refined by the router agent")
    thought_process: List[str] = Field(default_factory=list, description="Reasoning trail of the agents")
    response: Optional[str] = Field(default=None, description="Final response to the user")
    agent_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of agent interactions")
    requires_research: bool = Field(default=False, description="Whether more research is needed")
    research_topics: List[str] = Field(default_factory=list, description="Topics that need further research")

# Create a simple document store for demonstration
def create_vector_store(documents_path="./documents"):
    # Load documents (create this directory and add text files)
    documents = []
    
    # For demonstration, let's create a sample document if it doesn't exist
    if not os.path.exists(documents_path):
        os.makedirs(documents_path)
    
    sample_file = os.path.join(documents_path, "sample.txt")
    if not os.path.exists(sample_file):
        with open(sample_file, "w") as f:
            f.write("""
            Retrieval Augmented Generation (RAG) combines retrieval systems with generative AI models.
            RAG enhances output quality by grounding generation in relevant retrieved information.
            RAG helps reduce hallucinations by providing factual context to the model.
            LangGraph is a library for building stateful, multi-actor applications with LLMs.
            LangGraph allows you to create complex workflows and agent systems.
            """)
    
    # Load all documents in the directory
    for filename in os.listdir(documents_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_path, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    # Create vector store with your embeddings
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# 1. Router Agent - determines what information is needed
def router_agent(state: AgentState) -> AgentState:
    router_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a query router agent. Your job is to:
1. Analyze the user's query
2. Determine if it requires retrieval of information
3. Refine the query to make it more effective for retrieval
4. Decide if multiple sub-queries would be more effective
        
Be clear, concise, and helpful."""),
        HumanMessage(content="User Query: {query}")
    ])
    
    chain = router_prompt | llm | StrOutputParser()
    
    result = chain.invoke({"query": state.query})
    
    # Parse the router's output
    refined_query = state.query  # Default to original query
    requires_research = True  # Default to requiring research
    research_topics = []
    
    for line in result.split("\n"):
        if line.startswith("Refined Query:"):
            refined_query = line.replace("Refined Query:", "").strip()
        elif line.startswith("Sub-Queries:"):
            research_topics = [q.strip() for q in line.replace("Sub-Queries:", "").split(",")]
    
    # Update state
    state.refined_query = refined_query
    state.requires_research = requires_research
    state.research_topics = research_topics or [refined_query]
    state.thought_process.append(f"Router: {result}")
    state.agent_history.append({"agent": "router", "output": result})
    
    return state

# 2. Retrieval Agent - fetches relevant information
def retrieval_agent(state: AgentState) -> AgentState:
    # Initialize or get vector store
    vectorstore = create_vector_store()
    
    # Retrieve context for each research topic
    all_context = []
    for topic in state.research_topics:
        docs = vectorstore.similarity_search(topic, k=3)
        topic_context = [f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)]
        all_context.extend(topic_context)
    
    # Update state
    state.context = all_context
    state.thought_process.append(f"Retriever: Retrieved {len(all_context)} relevant documents")
    state.agent_history.append({"agent": "retriever", "output": f"Retrieved {len(all_context)} documents"})
    
    return state

# 3. Synthesis Agent - combines information and generates response
def synthesis_agent(state: AgentState) -> AgentState:
    synthesis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a synthesis agent. Your job is to create a comprehensive, accurate response based on:
1. The original user query
2. The retrieved context
        
Cite specific information from the context. Be concise but thorough."""),
        MessagesPlaceholder(variable_name="agent_messages"),
        HumanMessage(content="""
Original Query: {query}
Refined Query: {refined_query}

Retrieved Context:
{context}

Create a well-structured response that directly answers the query.
""")
    ])
    
    # Convert agent history to messages
    agent_messages = []
    for entry in state.agent_history:
        agent_messages.append(AIMessage(content=f"{entry['agent']}: {entry['output']}"))
    
    chain = synthesis_prompt | llm | StrOutputParser()
    
    context_text = "\n\n".join(state.context)
    result = chain.invoke({
        "query": state.query,
        "refined_query": state.refined_query,
        "context": context_text,
        "agent_messages": agent_messages
    })
    
    # Update state
    state.response = result
    state.thought_process.append(f"Synthesizer: Generated response based on {len(state.context)} documents")
    state.agent_history.append({"agent": "synthesizer", "output": result})
    
    return state

# 4. Critic Agent - evaluates the response for accuracy and completeness
def critic_agent(state: AgentState) -> AgentState:
    critic_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a critic agent. Your job is to:
1. Evaluate the response for accuracy, relevance, and completeness
2. Determine if more research is needed
3. Suggest improvements if necessary"""),
        HumanMessage(content="""
Original Query: {query}
Response to Evaluate: {response}
Available Context: {context_summary}

Evaluate this response and determine if it's ready to present to the user.
""")
    ])
    
    chain = critic_prompt | llm | StrOutputParser()
    
    # Summarize context for brevity
    context_summary = f"{len(state.context)} documents retrieved"
    
    result = chain.invoke({
        "query": state.query,
        "response": state.response,
        "context_summary": context_summary
    })
    
    # Parse critic's feedback
    needs_improvement = "more research needed" in result.lower() or "insufficient" in result.lower()
    
    # Update state
    state.thought_process.append(f"Critic: {result}")
    state.agent_history.append({"agent": "critic", "output": result})
    state.requires_research = needs_improvement
    
    return state

# Define routing logic
def router_decide(state: AgentState) -> str:
    """Decide next step based on the current state."""
    if state.requires_research and not state.context:
        # If research is needed and no context yet, do retrieval
        return "retriever"
    elif state.context and not state.response:
        # If we have context but no response yet, synthesize
        return "synthesizer"
    elif state.response and state.requires_research:
        # If we have a response but it needs improvement, get more context
        return "retriever"
    else:
        # We have a good response, end the process
        return END

# Build the graph
def build_rag_graph():
    # Define nodes
    nodes = {
        "router": router_agent,
        "retriever": retrieval_agent,
        "synthesizer": synthesis_agent,
        "critic": critic_agent
    }
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    for name, fn in nodes.items():
        workflow.add_node(name, fn)
    
    # Add edges
    workflow.add_edge("router", router_decide)
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "critic")
    workflow.add_edge("critic", router_decide)
    
    # Compile the graph - no checkpointer needed for the latest LangGraph
    app = workflow.compile()
    
    return app

# Main execution function
def run_multiagent_rag(query: str):
    """Run the multiagent RAG system with a user query."""
    # Build the graph
    graph = build_rag_graph()
    
    # Initialize state
    initial_state = AgentState(query=query)
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return {
        "query": result.query,
        "response": result.response,
        "thought_process": result.thought_process,
    }

# Example usage
if __name__ == "__main__":
    query = "How does RAG work with LangGraph?"
    result = run_multiagent_rag(query)
    print("\nFINAL RESPONSE:")
    print(result["response"])
    print("\nTHOUGHT PROCESS:")
    for thought in result["thought_process"]:
        print(f"- {thought}")