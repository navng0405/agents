from pathlib import Path
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

#1. Define the schema for our state
class AgentState(TypedDict):
    # The 'messages' key will store the conversation history.
    messages: Annotated[list, add_messages]

#2. Define the Tool
@tool
def get_weather(city: str):
    """Fetch the current weather for a specific city"""
    #In a real scenario, you would call an API. For now, we simulate logic.
    if "london" in city.lower():
        return "It's 15°C and cloudy in London."
    elif "paris" in city.lower():
        return "It's 20°C and sunny in Paris."
    else:
        return f"Sorry, I don't have weather data for {city}."
    
# 3. Setup the LLM & Tools --
tools = [get_weather]
model = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools(tools)

#4. Build the Graph
def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

#Define the graph
workflow = StateGraph(AgentState)

#Add nodes
workflow.add_node("call_model", call_model)
workflow.add_node("tool_node", ToolNode(tools))

#Define the flow
workflow.add_edge(START, "call_model")

#Logic: Does the LLM want to call a tool? If so, we go to the tool node. Otherwise, we end.
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END

workflow.add_conditional_edges("call_model", should_continue)

workflow.add_edge("tool_node", "call_model")

#Compile the graph into an executable app
app = workflow.compile()

def export_graph_diagram() -> Path:
    """Write the graph structure as Mermaid so the flow is easy to inspect."""
    diagram_path = Path(__file__).with_name("graph_flow.mmd")
    diagram_path.write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
    return diagram_path

#Exection
if __name__ == "__main__":
    diagram_path = export_graph_diagram()
    print(f"Graph diagram saved to: {diagram_path}")

    #Test with a query about the weather
    result = app.invoke({"messages": [HumanMessage(content="What's the weather like in London?")]})
    print("Result:", result)
