from agent.graph import build_graph
from langchain_core.messages import HumanMessage


def _normalize_user_input(user_input: str) -> str:
    stripped = user_input.strip()
    if "/" in stripped and " " not in stripped:
        return f"How many GitHub stars does the repository {stripped} have?"
    return stripped


def _print_agent_response(final_state: dict) -> None:
    messages = final_state.get("messages", [])
    if not messages:
        print("Agent: No response generated.")
        return

    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        ).strip()

    if content:
        print(f"Agent: {content}")
    else:
        print("Agent: I completed the request but returned an empty message.")


def main():
    print("Building Agentic Graph...")
    app = build_graph()

    print("\nAgent ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        prompt = _normalize_user_input(user_input)
        initial_state = {"messages": [HumanMessage(content=prompt)]}
        final_state = app.invoke(initial_state)
        _print_agent_response(final_state)


if __name__ == "__main__":
    main()
