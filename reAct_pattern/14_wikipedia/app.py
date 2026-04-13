import operator
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

#1. Define the schema for our state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# -- 2. Define the targeted tool --
@tool
def fetch_wikipedia_early_life(person_name: str) -> str:
    """Fetches the full Wikipedia page for a person and extracts ONLY the 'Early Life' section. If the person or section is not found, return an appropriate message."""
    wiki = WikipediaAPIWrapper()
    # Fetching the page content
    page_content = wiki.run(person_name)

    #Logic to target a specific section (Early Life)
    #Note : Real-world parsing misght use Regex, but for this agent,
    # we simulate the 'Targeted extraction' pattern,
    if "Early life" in page_content:
        #Simple split logic to simulate section extraction
        parts = page_content.split("Early life")
        return f"EARLY LIFE SECTION FOR {person_name.upper()}:\n{parts[1].split('==')[0].strip()}"
    return f"Could not find 'Early Life' section for {person_name}. Here's the full page content:\n{page_content}"

#3. The Summarizer Model
model = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools([fetch_wikipedia_early_life])

#--4.Logic nodes--
def call_summarizer(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

#5.Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("summarizer", call_summarizer)
workflow.add_node("tool_node", ToolNode([fetch_wikipedia_early_life]))

workflow.set_entry_point("summarizer")

def router(state: AgentState):
    print("Router checking if we should call the tool. Last message:", state["messages"][-1])
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END

workflow.add_conditional_edges("summarizer", router)
workflow.add_edge("tool_node", "summarizer")

app = workflow.compile()

#Execution
if __name__ == "__main__":
    query = "Tell me about the early life of Trisha Krishnan."
    inputs = {"messages": [HumanMessage(content=query)]}

    for chunk in app.stream(inputs, stream_mode="values"):
        msg = chunk["messages"][-1]
        role = "AI" if not isinstance(msg, HumanMessage) else "User"
        print(f"{role}: {msg.content if msg.content else 'Fetching Wikipedia...'}")