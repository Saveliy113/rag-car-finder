"""
Loaders module - Client initialization for Qdrant and OpenAI
"""
from .qdrant import get_qdrant_client, init_qdrant_client
from .openAi import get_openai_client, init_openai_client

__all__ = [
    "get_qdrant_client",
    "init_qdrant_client",
    "get_openai_client",
    "init_openai_client",
]
