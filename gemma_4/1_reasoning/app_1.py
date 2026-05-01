import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Load environment variables (API Key)
load_dotenv()

# --- 1. STATE DEFINITION ---
class State(TypedDict):
    # add_messages allows the graph to append new responses to the history
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. TOOLS ---
@tool
def verify_ad_compliance(copy_text: str) -> str:
    """Checks ad copy against corporate compliance. Use this before finalizing any output."""
    prohibited_keywords = ["guarantee", "always", "risk-free", "instant wealth"]
    found = [word for word in prohibited_keywords if word in copy_text.lower()]
    
    if found:
        return f"Compliance Check Failed: Prohibited terms found: {found}. Please rewrite."
    return "Compliance Check Passed: Copy is safe to use."

tools = [verify_ad_compliance]

# --- 3. MODEL INITIALIZATION ---
# Using the 31B flagship model for maximum reasoning performance
llm = ChatOllama(
    model="gemma4:e2b", # This is the mobile-optimized version
    temperature=0.1,
    base_url="http://localhost:11434"
)
llm_with_tools = llm.bind_tools(tools)

# --- 4. NODE LOGIC ---
def get_latest_user_text(state: State) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, list):
                return "\n".join(
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in content
                )
            return str(content or "")
    return ""


def route_request(state: State) -> str:
    user_text = get_latest_user_text(state).lower()
    if "image" in user_text or "screenshot" in user_text:
        return "vision"
    return "agent"


def agent_reasoning_node(state: State):
    """
    This node triggers Gemma 4's Thinking Mode.
    It parses the internal 'thought' channel separately from the final response.
    """
    # System instruction specifically for Gemma 4 native reasoning
    sys_msg = SystemMessage(
        content=(
            "<|think|> You are a Staff AI Engineer at a major retail tech company. "
            "Think through compliance constraints and technical requirements step-by-step. "
            "Use the tools provided if the user's request involves marketing copy."
        )
    )
    
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    
    # Optional: Logic to separate thought from response if you want to log it
    # Gemma 4 uses <|channel>thought and <channel|> tags
    content = response.content
    if isinstance(content, list):
        content_text = "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        content_text = str(content or "")

    if "<|channel>thought" in content_text and "<channel|>" in content_text:
        parts = content_text.split("<channel|>", 1)
        thought_process = parts[0].replace("<|channel>thought", "").strip()
        final_output = parts[1].strip()
        
        print("\n--- GEMMA 4 INTERNAL MONOLOGUE ---")
        print(thought_process)
        print("----------------------------------\n")
        
        # Update response content to show only the clean output to the next node
        response.content = final_output
    else:
        print("\n--- RAW MODEL RESPONSE ---")
        print(content_text or "<empty content>")
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            print(f"Tool calls: {tool_calls}")
        print("--------------------------\n")

    return {"messages": [response]}


def vision_analysis_node(state: State):
    """
    Handles image-oriented requests with a vision-specific system instruction.
    """
    sys_msg = SystemMessage(
        content=(
            "You are a multimodal analyst helping a user reason about an image or screenshot. "
            "If the user has not actually provided the image, clearly say that you need the "
            "image or a detailed description before you can inspect it. "
            "Use the tools provided if the request also involves marketing copy compliance."
        )
    )

    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# --- 5. GRAPH CONSTRUCTION ---
builder = StateGraph(State)

# Define Nodes
builder.add_node("agent", agent_reasoning_node)
builder.add_node("vision", vision_analysis_node)
builder.add_node("tools", ToolNode(tools=tools))

# Define Edges
builder.add_conditional_edges(
    START,
    route_request,
    {
        "agent": "agent",
        "vision": "vision",
    },
)

# The tools_condition edge checks if the model wants to call 'verify_ad_compliance'
builder.add_conditional_edges("agent", tools_condition)
builder.add_conditional_edges("vision", tools_condition)

# Return to the same node family after tool execution
builder.add_conditional_edges(
    "tools",
    route_request,
    {
        "agent": "agent",
        "vision": "vision",
    },
)

# Compile
app = builder.compile()

# --- 6. EXECUTION ---
if __name__ == "__main__":
    print("Gemma 4 + LangGraph Agent Initialized.")
    
    # Example prompt that forces tool usage and reasoning
    user_input = "Draft a short ad for our new electronics sale. Use the phrase: 'We guarantee the lowest prices!'"
    
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    for event in app.stream(initial_state, stream_mode="values"):
        # The last message in the list is the current step's output
        last_msg = event["messages"][-1]
        if last_msg.content:
            last_msg.pretty_print()
