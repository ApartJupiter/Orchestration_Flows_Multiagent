### Hierarchical Flow (Master-Slave Model) ###

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import langgraph
from langgraph.graph import StateGraph
from typing import TypedDict, Optional

# Initialize LLM
llm = ChatOpenAI(open_api_base="http://localhost:1234/v1", model="deepseek-r1-distill-qwen-7b", temperature=0, openai_api_key="fake-key")

# Define the state structure
class HierarchicalState(TypedDict):
    user_query: str
    order_status: Optional[str]
    refund_info: Optional[str]
    tech_support: Optional[str]
    escalation: Optional[str]

def order_status_agent(state: HierarchicalState):
    if "order" in state["user_query"].lower():
        return {"order_status": "Your order is out for delivery."}
    return {}

def refund_agent(state: HierarchicalState):
    if "refund" in state["user_query"].lower():
        return {"refund_info": "Refunds take 5-7 business days."}
    return {}

def tech_support_agent(state: HierarchicalState):
    if "issue" in state["user_query"].lower():
        return {"tech_support": "Try restarting your device."}
    return {}

def escalation_agent(state: HierarchicalState):
    if not any([state.get("order_status"), state.get("refund_info"), state.get("tech_support")]):
        return {"escalation": "Escalating to a human agent."}
    return {}

def master_agent(state: HierarchicalState):
    return ["order_status", "refund_agent", "tech_support", "escalation"]

workflow = StateGraph(HierarchicalState)
workflow.add_node("order_status", order_status_agent)
workflow.add_node("refund_agent", refund_agent)
workflow.add_node("tech_support", tech_support_agent)
workflow.add_node("escalation", escalation_agent)
workflow.add_conditional_edges("master_agent", master_agent)
workflow.set_entry_point("master_agent")

graph = workflow.compile()

# Test Execution
user_query = "Where is my order?"
result = graph.invoke({"user_query": user_query})
print("Order Status:", result.get("order_status", "N/A"))
print("Refund Info:", result.get("refund_info", "N/A"))
print("Tech Support:", result.get("tech_support", "N/A"))
print("Escalation:", result.get("escalation", "N/A"))
