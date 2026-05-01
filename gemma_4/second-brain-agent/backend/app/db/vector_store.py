from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from app.core.config import settings

embeddings = OllamaEmbeddings(model="gemma4:e2b", base_url="http://localhost:11434")


def get_vector_store():
    vector_store = PGVector(
        connection_string=settings.DATABASE_URL,
        collection_name="user_memories",
        embeddings=embeddings,
        use_json=True,  # Store metadata as JSON
    )
    return vector_store