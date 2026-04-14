import sqlite3
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver

#Start : The process begins. LangGraph looks at the config for a thread_id.
#Lookup : SqliteSaver queries the DB. If it finds architect_session_q, it loads the message history into the state.
# Agent node : the chat function receives the messages, call ollama, and returns a new message
# Reducer : the annotated list logic takes the new message and appends it to the existing list.
# Checkpoint: Before reaching END. the updated state us serialized and saved back to the sqlite db.

# 1. Define the State
# The state is the "schema" of the agent
class State(TypedDict):
    # We use a list of messages and annotate it so new messages are append messages: 
    # Annoted[list[BaseMessage], lambda x,y:x+y]
    messages: Annotated[list[BaseMessage], lambda x,y:x+y]

# 2. Initialize the LLM
# Ensure you have 'llama3' or similar pulled: 'ollama pull llama3'
llm = ChatOllama(model="llama3.1:8b", temperature=0)

#3. Define the Node (The logic)
def chatbot (state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

#4. Build the graph
workflow = StateGraph(State)
workflow.add_node("agent", chatbot)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

#5. Set up Persistence
# This creates a local SQLite D names 'checkpoints.sqlite'
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread = False)
memory = SqliteSaver(conn)

#Compile the graph with the checkpointer
app = workflow.compile(checkpointer = memory)

# ---- Execution Logic --
config = {"configurable": {"thread_id": "architect_session_1"}}

def run_agent(user_input: str):
    print(f"\nUser: {user_input}")
    events = app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values"
    )
    final_content = ""
    for event in events:
        if "messages" in event:
            last_message = event["messages"][-1]
            if hasattr(last_message, "content"):
                # Just printing the final response for clarity
                final_content = last_message.content
                #print(f'AI: {final_content}')

    print(f'AI: {final_content}')

if __name__ == "__main__":
    # Test 1: Tell the AI your name
    # run_agent("Hi,I am Naveen.  I'm building a library of 100 agents.")

    #Test 2: Ask the AI your name (Run this after restarting the script)
    run_agent("Do you remember what my name is and what i am building?")


