from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

# 1. Desine the Structured Output Schema
# This forces the LLM to pick one of the three options.
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the text")
    confidence: float = Field(description="Score between 0 and 1 indicating the confidence of the sentiment analysis")

# 2. Define our Hero State
class AgentState(TypedDict):
    user_text: str
    mood: str
    confidence_score: float

#3. Setup Ollama with Structured Output
#In 2026, 'json_schema' is the default and most reliable methos for local models to output structured data.
llm = ChatOllama(model="llama3.1:8b", temperature=0)
structured_output = llm.with_structured_output(SentimentAnalysis)

#4. Define the Reasoning Node
def sentiment_analysis_node(state: AgentState):
    print("---LOG : Analyzing Sentiment---")
    result = structured_output.invoke(state["user_text"])
    return {"mood": result.sentiment, "confidence_score": result.confidence}

#5. Build the Graph
builder = StateGraph(AgentState)
builder.add_node("sentiment_analysis", sentiment_analysis_node)

builder.add_edge(START, "sentiment_analysis")
builder.add_edge("sentiment_analysis", END) 

#6. Compile and Run
app = builder.compile()

#Test the agent
inputs = {"user_text": "I had a great day!"}
result = app.invoke(inputs)
print(f"Sentiment: {result['mood']}, Confidence: {result['confidence_score']:.2f}") 

inputs = {"user_text": "I'm absolutely frustrated with this slow service, but the food was decent."}
result = app.invoke(inputs)
print(f"Sentiment: {result['mood']}, Confidence: {result['confidence_score']:.2f}") 