import json
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END

#1. Define the schema for our state
class AgentState(TypedDict):
    input_query: str
    llm_response: str
    metadata: Dict[str, Any]
    step_count: int

#2. The Logging Agent ("The Observer")
def logger_node(state: AgentState):
    """
    A Side-effect Node: Prints the state for observability
    and returns an empty dict (passing state along unchanged).
    """

    print("\n" + "="*50)
    print("LOGGING AGENT: CURRENT STATE")
    print("="*50)

    #Using json.dumps for pretty printing the state
    print(json.dumps(state, indent=2))

    print("="*50 + "\n")

    # In Langgraph, returning an empty dict tells the graph
    # Don't change anything in the global state."
    return {}

#3. A Dummy Worker Node
def processor_node(state: AgentState):
    """Simulates some processing and updates the state."""
    new_response = f"Processed: {state['input_query']}"
    return {"llm_response": new_response, "step_count": state.get("step_count", 0) + 1} 
#4. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("logger", logger_node)
workflow.add_node("processor", processor_node)

workflow.add_edge(START, "processor")
workflow.add_edge("processor", "logger")
workflow.add_edge("logger", END)    

app =workflow.compile()

if __name__ == "__main__":
    #Test the agent with an initial state
    initial_state = {
        "input_query": "What is the weather like today?",
        "llm_response": "",
        "metadata": {"source": "user_input"},
        "step_count": 0
    }

    result = app.invoke(initial_state)
    print("Final State after processing:", json.dumps(result, indent=2))