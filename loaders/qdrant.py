import os
from qdrant_client import QdrantClient
from typing import Optional

# Singleton instance
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client singleton"""
    global _qdrant_client
    
    if _qdrant_client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        _qdrant_client = QdrantClient(host=host, port=port)
    
    return _qdrant_client


def init_qdrant_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """Initialize Qdrant client (called at startup)"""
    global _qdrant_client
    
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=host, port=port)
    
    return _qdrant_client