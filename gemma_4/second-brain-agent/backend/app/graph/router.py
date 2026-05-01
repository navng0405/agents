from langchain_ollama import ChatOllama
from app.graph.state import AgentState
from app.core.config import settings

router_llm = ChatOllama(model="gemma4:8b", temperature=0, base_url=settings.OLLAMA_BASE_URL)

async def router_node(state: AgentState):
    """Analyzes the user input and decides which action node to route to next."""
    last_message = state["messages"][-1].content
    system_prompt = f"""
      Catagorize this user message into ONE of these three intents:
      1. 'store': Use this is the user is providing information they want remembered for later.
      2. 'retrieve': Use this if the user is asking a question or searching for infrmation that would require accessing the stored memories.
      3. 'chat': Use this for general conversation that doesn't fit the above two categories.

        User Message: "{last_message}"
        Respond with ONLY the intent label: 'store', 'retrieve', or 'chat'.
    """
    response = await router_llm.ainvoke(system_prompt)
    decision = response.content.strip().lower()

    if decision == "store":
        return "store"
    elif decision == "retrieve":
        return "retrieve"
    else:
        return "chat"