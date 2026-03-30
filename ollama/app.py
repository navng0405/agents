from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm =ChatOllama(model="llama3.1:8b")

class State(TypedDict):
    messages: list
    response: str

def generate(state: State) -> str:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"response": response.content}

graph = StateGraph(State)
graph.add_node("generate", generate)
graph.set_entry_point("generate")
graph.add_edge("generate", END)

app = graph.compile()

initial_state = {"messages": [("user", "Why learning agentic AI (langgrapg) is very important for software enginner in 2026 and what is the future for siftware engineers")], "response": ""}
result= app.invoke(initial_state)
print(result["response"])