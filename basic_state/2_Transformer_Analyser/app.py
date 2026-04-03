import operator
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict

#Define the "State" (The memory of the agent)
class State(TypedDict):
    # Annotated with operator.add allows messages to append rather then overwrite
    user_input: str
    agent_output: str

#2. Initialize the LLM (The brain of the agent)
llm =ChatOllama(model="llama3.1:8b")

#3. Define the Node (The function that processes the state)
def echo_node(state: State):
    # In a real agent, you would do: response = llm.invoke(state["user_input"])
    # For a pure Echo , we just format the strin.
    print("---LOG : Node is procesing---")
    return {"agent_output": f"Echo: {state['user_input']}"}

# 4.Build the graph (The structure of the agent)
workflow = StateGraph(State)

#Add the node to the graph
workflow.add_node("echo", echo_node)

#Define the entry point and exit point
workflow.add_edge(START, "echo")
workflow.add_edge("echo", END)

# 5. Compile and Run
app = workflow.compile()

#Test the agent
inputs = {"user_input": "Hello, how are you?"}
result = app.invoke(inputs)
print(result["agent_output"])