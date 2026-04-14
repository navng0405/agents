import os
import operator
from pathlib import Path
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

#1. Define the schema for our state
class AgentState(TypedDict):
    # The 'messages' key will store the conversation history.
    messages: Annotated[list[BaseMessage], operator.add]

#2. Define the local I/O tools (with Safety Boundaries) --
# LEARNING : We define a strict absolute path for our sandbox"
#The agent will not be allowed to look outside of this specific directory, ensuring we maintain control over the file system access.
BASE_DATA_DIR = Path(__file__).parent / "data"

@tool
def list_files() -> str:
    """Lists all files in the data directory."""
    try:
        if not BASE_DATA_DIR.exists():
            return "Data directory does not exist."
        files = [f.name for f in BASE_DATA_DIR.iterdir() if f.is_file() and f.suffix in {".txt", ".csv"}]
        if not files:
            return "No .txt or .csv files found in the data directory."
        return "Files in data directory:\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"
    
@tool
def read_file(file_name: str) -> str:
    """Reads the content of a specific .txt file,
    Input should ONLY be filename (e.g.document.txt) , not the full path. 
    The agent will only be able to read files that are in the data directory."""
    try:
        #Securely construct the file path
        target_path = (BASE_DATA_DIR / file_name).resolve()

        #Check if the target path is within the allowed directory
        if not str(target_path).startswith(str(BASE_DATA_DIR.resolve())):
            return "Access denied: You can only read files within the data directory."
        if not target_path.exists() or target_path.suffix != ".txt":
            return f"File '{file_name}' not found in data directory or is not a .txt file."
        with open(target_path, "r", encoding="utf-8") as f:
            #LEARNING : We truncate the content to a reasonable length to prevent overwhelming the model and to simulate real-world constraints.
            content = f.read(2000)  # Read only the first 2000 characters
            return f"Content of {file_name}:\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"
    
#-- 3. The System-Aware Model setup --
tools = [list_files, read_file]
model = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools(tools)

#4. Logic nodes
def call_model(state: AgentState):
    #Learning :We let the LLM figure out the multi-step process on its own, which is a key aspect of the ReAct pattern. The model can decide when to list files and when to read a specific file based on the conversation context.
    response = model.invoke(state["messages"])
    return {"messages": [response]}

#5.Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("io_tools", ToolNode(tools))

workflow.set_entry_point("agent")

def router(state: AgentState):
    print("Router checking if we should call the tool. Last message:", state["messages"][-1])
    if state["messages"][-1].tool_calls:
        return "io_tools"
    return END

workflow.add_conditional_edges("agent", router)
workflow.add_edge("io_tools", "agent")

app = workflow.compile()

#Execution
if __name__ == "__main__":
    # Ensure the directory exisits for the test
    BASE_DATA_DIR.mkdir(exist_ok=True)
    with open(BASE_DATA_DIR / "example.txt", "w", encoding="utf-8") as f:
        f.write("This is an example file for the Local File Reader agent. It contains some sample text data that the agent can read and use to answer questions.")
    query = "What files do I have in my data directory? If there's a file named 'example.txt', read its content and summarize it for me."
    inputs = {"messages": [HumanMessage(content=query)]}
    
    for chunk in app.stream(inputs, stream_mode="values"):
        msg = chunk["messages"][-1]
        print(f"---{type(msg).__name__}---: {msg.content if msg.content else 'Processing...'}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Calling Tools: {[tc['name'] for tc in msg.tool_calls]}")
        elif msg.content:
            print(f"Model Response: {msg.content}")