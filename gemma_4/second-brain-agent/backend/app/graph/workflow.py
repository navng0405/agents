def run_second_brain_workflow(query: str) -> dict[str, object]:
    normalized = query.strip().lower()

    if any(keyword in normalized for keyword in ("remember", "save", "store", "note")):
        intent = "store"
    elif any(
        keyword in normalized for keyword in ("what", "how", "why", "plan", "help", "?")
    ):
        intent = "retrieve"
    else:
        intent = "chat"

    return {
        "answer": (
            "Intent detected: "
            f"{intent}. This is a lightweight fallback workflow response."
        ),
        "intent": intent,
        "retrieved_notes": [],
        "workflow_steps": ["classify_query", "generate_response"],
    }
