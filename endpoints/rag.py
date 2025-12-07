from fastapi import APIRouter, HTTPException
from models.models import RagQueryRequest
from loaders import get_qdrant_client, get_openai_client

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/search")
async def search_cars(request: RagQueryRequest):
    """
    Search for cars using RAG
    Example: Search for "white Toyota Camry under 10 million"
    """
    try:
        # Get clients (already initialized at startup)
        qdrant_client = get_qdrant_client()
        openai_client = get_openai_client()
        
        # Create embedding for the query
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.question
        )
        query_embedding = response.data[0].embedding
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name="cars",
            query_vector=query_embedding,
            limit=request.top_k
        )
        
        # Format results
        results = []
        for result in search_results:
            car_data = result.payload.copy()
            car_data['similarity_score'] = result.score
            car_data['id'] = result.id
            results.append(car_data)
        
        return {
            "query": request.question,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

