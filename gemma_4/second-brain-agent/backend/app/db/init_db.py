from sqlalchemy import text
from app.db.session import engine

def init_db():
    with engine.connect() as connection:
        # Enable pgvector in the current database.
        connection.execute(
            text(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """
            )
        )
        connection.commit()
        print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
