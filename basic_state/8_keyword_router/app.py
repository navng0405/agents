from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

#1. Define the schema for our state
class AgentState(TypedDict):
    input_query: str
    llm_response: str
    classification: str

#2. The Destination Nodes --
def questions_node(state: AgentState):
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    # Tell the LLM specifically how to handle a question
    prompt = f"The user asked a question: '{state['input_query']}'. Classify this as 'question' and provide a brief answer."
    response = llm.invoke(prompt)
    return {"llm_response": response.content, "classification": "question"}

def statement_node(state: AgentState):
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    # Tell the LLM specifically how to handle a statement
    prompt = f"The user made a statement: '{state['input_query']}'. Classify this as 'statement' and provide a brief acknowledgment."
    response = llm.invoke(prompt)
    return {"llm_response": response.content, "classification": "statement"}

#3. The Router Node
def router_node(state: AgentState):
    """
    The Decision Logic.
    This function returns the name of the next node to visit."""

    if "???" in state["input_query"]:
        return "handle_question"
    else:
        return "handle_statement"
    
#4. Build the Graph
workflow = StateGraph(AgentState)

#Ad our processing nodes
workflow.add_node("handle_question", questions_node)
workflow.add_node("handle_statement", statement_node)

#Define the Conditional Entry point
# We use START -> router logic to decide where to go first
workflow.add_conditional_edges(START, router_node, {
    "handle_question": "handle_question",
    "handle_statement": "handle_statement"})

#Both paths lead to the end
workflow.add_edge("handle_question", END)
workflow.add_edge("handle_statement", END)

app = workflow.compile()

#Exection
if __name__ == "__main__":
    #Test with a question
    result = app.invoke({"input_query": "What is the capital of France???"})
    print("Result for Question:", result)

    #Test with a statement
    result = app.invoke({"input_query": "I love programming."})
    print("Result for Statement:", result)
