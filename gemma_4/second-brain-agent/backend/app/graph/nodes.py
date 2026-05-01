from langchain_ollama import ChatOllama
from app.graph.state import AgentState
from langchain_core.documents import Document
from app.db.vector_store import get_vector_store

llm = ChatOllama(model="gemma4:e2b", temperature=0.8, base_url="http://localhost:11434")

async def response_node(state: AgentState):
    """This node generates a response based on the current conversation history."""
    print("Generating response with Gemma 4...")
    memories_str = "\n".join([f"- {m}" for m in state["retrieved_memories"]])

    system_prompt = f"""<|think|>
    You are a Digital second brain. Use the following context to help the user. Always think step-by-step and be concise.
    Context:
    {memories_str}

     REASONING STEPS:

     1. Analyze the user's query and the context provided by the retrieved memories are relevent to the user's query.
     2. If irrelevant, search for a general answer but mention you didn't find relevant information in the memories.
     3. Formulate a response that feels personalized and helpful, as if you are a trusted assistant who has access to the user's past information but is not limited by it. Always be concise and to the point.
    """

    messages = [("system", system_prompt)] + state["messages"]
    response = await llm.ainvoke(messages)
    return {
        "messages": [response], "current_action": "responding"

    }


async def store_momory_node(state: AgentState):
    """This node stores the latest user query and the agent's response as a new memory."""
    print("Storing new memory...")
    last_message = state["messages"][-1].content
    user_id = state["user_profile"].user_id

    doc = Document(page_content=last_message, metadata={"user_id": user_id, "source":"chat_interaction", "type":"personal_fact"})
    vector_store = get_vector_store()

    await vector_store.add_documents([doc])

    return {
        "messages": [{"role":"assistant", "content":"I have added that to your second brain. I will remember it for next time!"}],
        "current_action": "storing_memory"
    }