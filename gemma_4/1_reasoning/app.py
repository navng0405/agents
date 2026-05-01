from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# 1. Define the State
# LangGraph relies on a state object passed between nodes. 
# `add_messages` ensures new messages are appended rather than overwritten.
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Define the Tools
# This is a mocked tool that the agent can route to if it needs to verify copy.
@tool
def check_ad_compliance(copy_text: str) -> str:
    """Always use this tool to check if the ad copy meets standard compliance guidelines before finalizing."""
    if "guarantee" in copy_text.lower():
        return "Compliance Alert: You cannot use the word 'guarantee'. Please revise to be less absolute."
    return "Compliance Passed: The copy is safe to publish."

tools = [check_ad_compliance]

# 3. Initialize Gemma 4
# We bind the tools directly to the model so Gemma 4 knows what functions it can invoke.
# Change this line in your app.py
llm = ChatOllama(
    model="gemma4:e2b", # This is the mobile-optimized version
    temperature=0.1,
    base_url="http://localhost:11434"
)
llm_with_tools = llm.bind_tools(tools)

# 4. Define the Nodes
# The agent node simply invokes Gemma 4 with the current conversation history.
def agent_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 5. Build the Graph
graph_builder = StateGraph(State)

# Add our core nodes: the LLM reasoning node and the tool execution node
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Define the flow
graph_builder.add_edge(START, "agent")

# The conditional edge checks if Gemma 4 decided to call a tool. 
# If yes -> route to "tools". If no -> route to END.
graph_builder.add_conditional_edges("agent", tools_condition)

# After a tool executes, return to the agent so it can read the tool's output
graph_builder.add_edge("tools", "agent")

# Compile into a runnable application
app = graph_builder.compile()

# 6. Execute the Agent
if __name__ == "__main__":
    test_copy = "Buy our new product! We guarantee it will increase your ROI by 50%!"
    
    print(f"Original Brief: {test_copy}\n")
    print("--- Agent Execution Trace ---")
    
    inputs = {"messages": [HumanMessage(content=f"Review and finalize this ad copy: '{test_copy}'")]}
    
    # Stream the events to see Gemma 4's reasoning and tool routing in action
    for event in app.stream(inputs, stream_mode="values"):
        last_message = event["messages"][-1]
        last_message.pretty_print()