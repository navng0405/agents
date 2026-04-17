import sqlite3
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Annotated, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver

#1. State Definition
class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    recalled_fact: str # A specific slot for our "look back" logic

#2. Setup llm
llm = ChatOllama(model="llama3.1:8b", temperature=0)

#3.The Recall node
# This is the brain looking at its own memory bank
def recall_history_node(state: State, config: RunnableConfig):
    #We access the history using the thread_id from the config
    #'limit=10' looks back at the last 10 versions of the state
    history = list(app.get_state_history(config))

    # Logic : Go back 5 versions if they exisit:
    #Note: Each interactions usually creates 2 states (Start of node, end of node)
    target_index = 5

    if len(history) > target_index:
        past_state = history[target_index]
        #Acessing the messages from that specific point in time
        past_messages = past_state.values.get("messages", [])
        if past_messages:
            fact = f"5 steps ago, the last messages was: {past_messages[-1].content}"
        else:
            fact = "I remember the time, but the messages are blurry."
    else:
        fact= "My memory doesn't go back that far yet"

    return {"recalled_fact": fact}

#4. The response node
def chatbot(state: State):
    prompt = (
        f"The user asked: {state['messages'][-1].content}.\n"
        f"Context from my memory: {state['recalled_fact']}"
    )
    response = llm.invoke(prompt)
    return {"messages": [response]}

#5 Graph Construction
workflow = StateGraph(State)

workflow.add_node("recall_fact", recall_history_node)
workflow.add_node("agent", chatbot)

workflow.add_edge(START, "recall_fact")
workflow.add_edge("recall_fact", "agent")
workflow.add_edge("agent", END)

#Persistence Setup
conn = sqlite3.connect("recall_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)

#Mentor's Simulation
config = {"configurable":{"thread_id":"architect_002"}}

