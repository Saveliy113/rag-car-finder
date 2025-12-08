from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.include_router(health_router)
app.include_router(rag_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)