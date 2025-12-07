from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from endpoints.health import router as health_router
from endpoints.rag import router as rag_router
from loaders import init_qdrant_client, init_openai_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients at startup"""
    # Startup
    print("🚀 Initializing clients...")
    try:
        init_qdrant_client()
        print("✓ Qdrant client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Qdrant client: {e}")
        raise
    
    try:
        init_openai_client()
        print("✓ OpenAI client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize OpenAI client: {e}")
        raise
    
    print("✅ All clients initialized successfully")
    yield
    # Shutdown (if needed)
    print("🛑 Shutting down...")


app = FastAPI(
    title="RAG API",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(health_router)
app.include_router(rag_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)