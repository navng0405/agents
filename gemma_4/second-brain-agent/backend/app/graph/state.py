from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from app.api.schemas import UserProfile

class AgentState(TypedDict):
    """
    The state of the digital brain agent. This state will be passed between nodes in the graph.
    """
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation history, with automatic appending
    user_profile: Optional[UserProfile]  # User profile information, can be updated by a node that fetches or modifies user data

    retrived_memories: List[str]  # A slot to store any relevant memories retrieved during the agent's reasoning process. This can be used to provide context to the LLM or for decision-making.

    is_deletion_pending: bool  # A flag to indicate if the agent is currently in the process of deleting information, which can be used to prevent certain actions or trigger specific behaviors during that process.

    current_action: Optional[str]  # A slot to store the current action the agent is taking, which can be useful for debugging, logging, or conditional logic in the graph.
