from typing import TypedDict, Annotated
import operator
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

#1. Define the schema for our state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

#2. Define the search tool
# DuckDuckGo Search API can be integrated here through LangChain's community tool.
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

#-- 3. The researcher model
# We use llama here, but you can choose any model that supports tool calling.
model = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools(tools)

#4.Logic nodes
def call_researcher(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

#5.Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("researcher", call_researcher)
workflow.add_node("search_node", ToolNode(tools))



def should_search(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "search_node"
    return END

workflow.set_entry_point("researcher")
workflow.add_conditional_edges("researcher", should_search)
workflow.add_edge("search_node", "researcher")

app = workflow.compile()
 
#Execution
if __name__ == "__main__":
    #In a real scenario, the user would provide a query that requires research. For this demo, we hardcode it.
    query = "which stock had the highest growth in 2026?"
    inputs = {"messages": [HumanMessage(content=query)]}

    for chunk in app.stream(inputs, stream_mode="values"):
        msg = chunk["messages"][-1]
        role = "AI" if not isinstance(msg, HumanMessage) else "User"
        print(f"{role}: {msg.content if msg.content else 'Checking the web..'}")


