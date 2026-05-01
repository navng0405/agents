import contextlib
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from app.core.config import settings

DB_URL = settings.DATABASE_URL

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

@contextlib.contextmanager
def get_checkpointer():
    """
    Context manager to provide a PostgresSaver instance with a connection pool.
    """
    with ConnectionPool.connect(DB_URL, **connection_kwargs) as conn:
        with conn.connection() as connection:
            checkpointer = PostgresSaver(connection)
            checkpointer.setup()  # Ensure the necessary table is created
            yield checkpointer