import operator
from typing import Annotated, TypedDict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- 1. Schema Defination ---
class AgentState(TypedDict):
    #History of the conversation
    messages: Annotated[List[BaseMessage], operator.add]
    # The raw string we received from the previous node
    raw_content: str
    #The final polished output
    formatted_output: str

# --- 2. The Stylist Agent ---
class StylistAgent:
    def __init__(self):
        # Initialize the LLM (The brain of the agent)
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0.2)

        self.system_instruction = ("You are a professional technical editor. Your task is to take 'Raw Input'"
        "and polish it into a more formal and structured format. Focus on clarity, grammar, and conciseness while maintaining the original meaning."
        "\n\nRules: \n"
        "1. Preserve the original meaning of the input.\n"
        "2. Improve the overall readability and flow of the text.\n"
        "3. Use proper grammar and punctuation."
        "4. Avoid adding any new information that is not present in the original input.\n"
        "5. Format the output in a clear and organized manner, using paragraphs if necessary.\n\n"
        "Example:\n"
        "Raw Input: 'the quick brown fox jumps over the lazy dog'\n"
        "Formatted Output: 'The quick brown fox jumps over the lazy dog.'\n\n"
        "Raw Input: 'i have a meeting at 3pm, can you remind me?'\n"
        "Formatted Output: 'I have a meeting at 3 PM. Can you remind me?'")
        
    def format_text(self, state: AgentState):
        """Transforms the raw content into a polished format."""
        raw_text = state.get("raw_content", "")
        messages = state.get("messages", [])

        #If raw_content is empty, check the last message instead
        if not raw_text and messages:
            raw_text = messages[-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_instruction),
            ("human", "Raw Input: {input}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"input": raw_text})

        return {"formatted_output": response.content}
    
# --- 3. Build the Graph ---
def build_stylist_graph():
    #Initialize our agent logic
    stylist_agent = StylistAgent()

    #Define the graph
    workflow = StateGraph(AgentState)

    #Add the single node for this specification
    workflow.add_node("styling", stylist_agent.format_text)

    #Define the flow
    workflow.set_entry_point("styling")
    workflow.add_edge("styling", END)

    return workflow.compile()

# --- 4. Run the Agent ---
if __name__ == "__main__":
    app = build_stylist_graph()

    #Test the agent with different inputs
    test_inputs = [
        {"raw_content": "I did updated the file last eveneing and i send you the linkas asap"},
        {"raw_content": "i have a meeting at 3pm, can you remind me?"},
        {"raw_content": "this is an example of a poorly written sentence that needs to be polished."}
    ]

    for idx, input_state in enumerate(test_inputs):
        result = app.invoke(input_state)
        print(f"Test Case {idx + 1}:\nRaw Input: {input_state['raw_content']}\nFormatted Output: {result['formatted_output']}\n")
