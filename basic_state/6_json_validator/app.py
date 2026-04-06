import json
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

#1. Define the schema for our state
class AgentState(TypedDict):
    # The raw JSON string we want to validate
    raw_response: str
    parsed_json: Dict[str, Any]
    is_valid: bool
    error_message: str


def strip_code_fences(raw_text: str) -> str:
    """Remove Markdown code fences when the model wraps JSON in ``` blocks."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned

#2. The LLM Node --
def call_ollama(state: AgentState):
    print("call_ollama input:", state)
    """Simulates an agent task that is supposed to return JSON."""
    llm = ChatOllama(model="llama3.1:8b", temperature=0)

    prompt = ("Generate a JSON object representing a user profile with 'name' and 'age' fields."
              "Return ONLY the JSON. No preamble")
    response = llm.invoke(prompt)
    # Extract the text from the AIMessage object
    if hasattr(response, "content"):
        raw_response = response.content
    else:
        raw_response = str(response)
    
    print("call_ollama output:", {"raw_response": raw_response})
    return {"raw_response": raw_response}

#3.The Guardrail Node (JSON Validation)
def json_validator(state: AgentState):
    print("json_validator input:", state)
    """The core logic for data reliability"""
    raw_data = strip_code_fences(state.get("raw_response", ""))
    
    try:
        #Attempt to parse
        cleaned_data = json.loads(raw_data)
        print("json_validator output:", {"parsed_json": cleaned_data})
        return {
            "parsed_json": cleaned_data,
            "is_valid": True,
            "error_message": ""
        }
        
    except ValueError as e:
        #If parsing fails, mark as invalid and capture the error
        return {"is_valid": False, "error_message": str(e), "parsed_json": {}}
    
#4. Define the Graph
workflow = StateGraph(AgentState)

#Add Nodes
workflow.add_node("call_ollama", call_ollama)
workflow.add_node("json_validator", json_validator)

#Define the edges
workflow.add_edge(START, "call_ollama")
workflow.add_edge("call_ollama", "json_validator")
    
#for this spec, we terminate, but we could routr back to 'llm_call' for retry 
workflow.add_edge("json_validator", END)

#Compile the graph
app = workflow.compile()

#--Execution
if __name__ == "__main__":
    result = app.invoke({})
    print("Raw response:", result["raw_response"])
    print("Parsed JSON:", result["parsed_json"])
    print("Is valid:", result["is_valid"])
    if not result["is_valid"]:
        print("Error message:", result["error_message"])
