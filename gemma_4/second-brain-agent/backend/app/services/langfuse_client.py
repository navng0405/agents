from functools import lru_cache

from langfuse import Langfuse

from app.core.config import settings


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    has_credentials = bool(
        settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY
    )
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY or None,
        secret_key=settings.LANGFUSE_SECRET_KEY or None,
        host=settings.LANGFUSE_HOST,
        enabled=has_credentials,
    )
