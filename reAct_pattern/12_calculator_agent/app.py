import operator

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

#1. Define the schema for our state
class AgentState(TypedDict):
    # The 'operator.add' allows the state to append messages instead of overwriting them
    messages: Annotated[list[BaseMessage], operator.add]

#2. Define the Tool
@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    Input should be a string like '2*(3 + 4)'.
    supports +, - ,*,/ and parentheses.
    """
    try:
        # We use a restricted dict for basic safety in this local demo
        allowed_names = {"__builtins__": None}
        return str(eval(expression, allowed_names, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}" 
#3.The Strict LLM setup
# System Message is key: it acts as a guardrail to ensure the model uses the tool correctly.
system_message = ("You're a helpful calculator agent. Only use the 'calculate' tool to answer questions. "
 "If you need to perform a calculation, call the tool with the expression you want to evaluate. "
 "For example, if asked 'What is 2 + 2?', you should respond with 'calculate(2 + 2)'. "
 "Do not provide direct answers without using the tool.")
model = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools([calculate])

#4. Logic nodes
def call_model(state: AgentState):
    print("Calling model with messages:", state["messages"])
    # Inject the system prompt at the start of the conversation if not present.
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) and msg.content == system_message for msg in messages):
        messages = [SystemMessage(content=system_message)] + messages
    response = model.invoke(messages)
    return {"messages": [response]}

#5.Build the Graph
workflow = StateGraph(AgentState)

#Add nodes
workflow.add_node("call_model", call_model)
workflow.add_node("tool_node", ToolNode([calculate]))

#Define the flow
workflow.add_edge(START, "call_model")

#Logic: Does the LLM want to call a tool? If so, we go to the tool node. Otherwise, we end.
def should_continue(state: AgentState):
    print("Checking if we should continue to tool node. Last message:", state["messages"][-1])
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tool_node"
    return END

# Add conditional edges based on the LLM's response
workflow.add_conditional_edges("call_model", should_continue)
# If the tool was called, we need to go back to the model to process the tool's output and decide next steps.
workflow.add_edge("tool_node", "call_model")
    
#Compile the graph into an executable app
app = workflow.compile()

#Execution
if __name__ == "__main__":
    query = "What is k here? k+k/2 = 5, solve for k."
    inputs = {"messages": [HumanMessage(content=query)]}

    for output in app.stream(inputs, stream_mode="values"):
        last_msg = output["messages"][-1]
        print(f"[{type(last_msg).__name__}]: {last_msg.content if last_msg.content else 'Tool call made.'}")
