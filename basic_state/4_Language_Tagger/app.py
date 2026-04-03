import operator
from typing import Annotated, TypedDict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# --- 1. Schema Defination ---
# As an architect, always define your data contracts first.
class LanguageTag(BaseModel):
    """Identifying the langauage of the input text."""
    iso_code: str = Field(description="The ISO 639-1 language code, e.g., 'en' for English, 'es' for Spanish.")
    confidence: float = Field(description="A score between 0 and 1 indicating the confidence of the language identification.")  

# --- 2. State Defination ---
class AgentState(TypedDict):
    # We use Annotated with operator.add to allow the agent to append new tags rather than overwrite them.
    messages: Annotated[List[BaseMessage], operator.add]
    language_metadata: LanguageTag

# --- 2. The Language Detaction Agent ---
class LanguageDetectionAgent:
    def __init__(self):
        # Initialize the LLM (The brain of the agent)
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0)
        self.structured_output = self.llm.with_structured_output(LanguageTag)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a language detection assistant. Your task is to identify the language of the user's input text and provide a confidence score."),
            ("human", "{input_text}")
        ])
    
    def identify(self, state:AgentState):
        """Processes the last message and tags it with a language code."""
        last_message = state["messages"][-1].content

        #Invoke the structured chain
        chain = self.prompt | self.structured_output
        tagging_result = chain.invoke({"input_text": last_message})

        return {"language_metadata": tagging_result}
    
# --- 3. Build the Graph ---
def build_tagging_graph():
    #Initialize our agent logic
    identifier = LanguageDetectionAgent()

    #Define the graph
    workflow = StateGraph(AgentState)

    #Add the single node for this specification
    workflow.add_node("language_tagging", identifier.identify)

    #Define the flow
    workflow.set_entry_point("language_tagging")
    workflow.add_edge("language_tagging", END)

    return workflow.compile()

# --- 4. Run the Agent ---
if __name__ == "__main__":
    app = build_tagging_graph()

    #Test the agent with different inputs
    test_inputs = [
        {"messages": [HumanMessage(content="Hello, how are you?")]},
        {"messages": [HumanMessage(content="¡Hola! ¿Cómo estás?")]},
        {"messages": [HumanMessage(content="Bonjour, comment ça va?")]}
    ]

    for idx, input_state in enumerate(test_inputs):
        result = app.invoke(input_state)
        print(f"Test Case {idx+1}: Language: {result['language_metadata'].iso_code}, Confidence: {result['language_metadata'].confidence:.2f}")