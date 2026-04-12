import operator
from typing import Annotated, TypedDict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

#1. Define the schema for our state
# We add 'user_name' to the state. This persistes throughtout the graph run.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_name: Optional[str]

#2. The Logic Nodes --
def check_identify(state: AgentState):
    """
    Look at the state. If we don't have a name, ask the LLM to get it.
    If we do, greet them warmly.
    """
    
    last_message= state["messages"][-1].content.lower()

    # Simple logic: if the user just gave us their name, extract it. Otherwise, greet them.
    if "my name is" in last_message:
        # Extract the name (this is a very naive extraction, just for demo purposes)
        name = last_message.split("my name is")[-1].strip().split()[0]
        return {"user_name": name}
    return {}

def call_model(state: AgentState):
    prompt = "You are a friendly assistant."

    #Conditional logic based on state
    if state.get("user_name"):
        prompt += f" The user's name is {state['user_name']}. Greet them warmly."
    else:
        prompt += " The user has not told you their name yet. Ask them for their name."
       
    #We pass the instruction as a system message, and the conversation history as messages
    system_message = [AIMessage(content=prompt)]
    response = model.invoke(system_message + state["messages"])
    return {"messages": [response]}

#3. Build the Graph
model = ChatOllama(model="llama3.1:8b", temperature=0.7)
workflow =StateGraph(AgentState)

workflow.add_node("check_identify", check_identify)
workflow.add_node("call_model", call_model)

# Define the flow
workflow.add_edge(START, "check_identify")
workflow.add_edge("check_identify", "call_model")
workflow.add_edge("call_model", END)

app = workflow.compile()

#Exection
if __name__ == "__main__":
    #Test with a user who doesn't give their name
    result = app.invoke({"messages": [HumanMessage(content="Hello!")]})
    print("Result without name:", result)

    #Test with a user who gives their name
    result = app.invoke({"messages": [HumanMessage(content="Hello! My name is Alice.")]})
    print("Result with name:", result)