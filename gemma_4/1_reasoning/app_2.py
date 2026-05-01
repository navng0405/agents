import os
import base64
import mimetypes
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Load environment variables (API Key)
load_dotenv()

IMAGE_PATH = "/Users/preenav/Desktop/strip_test.jpg"


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
llm = ChatOllama(
    model="gemma4:e2b",
    temperature=0.1,
    base_url="http://localhost:11434",
)
llm_with_tools = llm.bind_tools(tools)


# --- 4. NODE LOGIC ---
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"


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
    if (
        ("tablet" in user_text or "tablets" in user_text)
        and ("count" in user_text or "how many" in user_text)
        and ("strip" in user_text or "image" in user_text or "photo" in user_text)
    ):
        return "tablet_count"
    if "image" in user_text or "screenshot" in user_text:
        return "vision"
    return "agent"


def agent_reasoning_node(state: State):
    """
    Handles standard text-first requests.
    """
    sys_msg = SystemMessage(
        content=(
            "<|think|> You are a Staff AI Engineer at a major retail tech company. "
            "Think through compliance constraints and technical requirements step-by-step. "
            "Use the tools provided if the user's request involves marketing copy."
        )
    )

    response = llm_with_tools.invoke([sys_msg] + state["messages"])

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


def tablet_counting_node(state: State):
    """
    Counts tablets in the configured strip image when the user asks for it explicitly.
    """
    if not os.path.exists(IMAGE_PATH):
        raise SystemExit(f"Image file not found: {IMAGE_PATH}")

    base64_image = image_to_base64(IMAGE_PATH)
    mime_type = get_image_mime_type(IMAGE_PATH)
    user_text = get_latest_user_text(state) or "Count the tablets in this strip image."
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            },
            {
                "type": "text",
                "text": (
                    f"User request: {user_text}\n"
                    "You are counting tablets in a blister strip image. Focus only on the blister pack. "
                    "Do not describe unrelated objects or invent animals, people, or background details. "
                    "Count visible tablet slots carefully, estimate only from what is clearly visible, "
                    "and return only compact JSON with keys total_slots, tablets_remaining, empty_slots, and notes."
                ),
            },
        ]
    )

    response = llm.invoke([message])
    return {"messages": [response]}


# --- 5. GRAPH CONSTRUCTION ---
builder = StateGraph(State)

builder.add_node("agent", agent_reasoning_node)
builder.add_node("vision", vision_analysis_node)
builder.add_node("tablet_count", tablet_counting_node)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_conditional_edges(
    START,
    route_request,
    {
        "agent": "agent",
        "tablet_count": "tablet_count",
        "vision": "vision",
    },
)

builder.add_conditional_edges("agent", tools_condition)
builder.add_conditional_edges("vision", tools_condition)
builder.add_edge("tablet_count", END)

builder.add_conditional_edges(
    "tools",
    route_request,
    {
        "agent": "agent",
        "tablet_count": "tablet_count",
        "vision": "vision",
    },
)

app = builder.compile()


# --- 6. EXECUTION ---
if __name__ == "__main__":
    print("Gemma 4 + LangGraph Agent Initialized.")
    if not os.path.exists(IMAGE_PATH):
        raise SystemExit(f"Image file not found: {IMAGE_PATH}")

    initial_state = {
        "messages": [
            HumanMessage(content="Count the tablets in this strip image.")
        ]
    }

    for event in app.stream(initial_state, stream_mode="values"):
        last_msg = event["messages"][-1]
        if last_msg.content:
            last_msg.pretty_print()
