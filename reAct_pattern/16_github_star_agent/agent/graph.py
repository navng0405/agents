from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from agent.state import AgentState
from core.config import MODEL_NAME, OLLAMA_BASE_URL
from agent.tools import get_github_stars


def build_graph() :
    # Initialize the LLM
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
    
    # Bind tools to the LLM
    tools = [get_github_stars]
    llm_with_tools = llm.bind_tools(tools)

    # Define the graph
    workflow = StateGraph(AgentState)

    # Node 1: Call the LLM
    def call_model(state: AgentState):
        print("Invoking LLM with messages:", state["messages"])
        response = llm_with_tools.invoke(state["messages"])
        print("LLM response:", response)
        return {"messages": [response]}

    workflow.add_node("call_model", call_model)

    # Node 2: Tool Node for GitHub Stars
    workflow.add_node("tool_node", ToolNode(tools))

    # Define edges
    workflow.add_edge(START, "call_model")

    # Conditional edge: If the LLM wants to call a tool, go to the tool node. Otherwise, end.
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tool_node"
        return END

    workflow.add_conditional_edges("call_model", should_continue)
    
    workflow.add_edge("tool_node", "call_model")

    return workflow.compile()