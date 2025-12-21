import json
import os
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from utils.logger import log, log_error
from utils.openai_queries import (
    extract_filters_from_query,
    create_embedding,
    generate_recommendation_response,
    detect_query_type
)
from utils.rag_filters import (
    build_qdrant_filter,
    sort_results_by_year_preference,
    calculate_dynamic_similarity_threshold
)
from utils.synonyms import normalize_filters_to_canonical
from loaders import get_qdrant_client, get_openai_client
from models.models import RagQueryRequest, RagQueryResponse

# Load environment variables
load_dotenv()

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/search", response_model=RagQueryResponse)
async def search_cars(request: RagQueryRequest):
    """
    Search for cars using RAG (Retrieval-Augmented Generation)
    """
    try:
        # Get clients
        qdrant_client = get_qdrant_client()
        openai_client = get_openai_client()

        # Load configuration from environment variables
        base_similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
        min_similarity_threshold = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.3"))
        filter_threshold_increment = float(os.getenv("FILTER_THRESHOLD_INCREMENT", "0.05"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        chat_temperature = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
        collection_name = os.getenv("COLLECTION_NAME", "cars")

        log(f"Processing query: {request.question}")

        # Detecting query type - general question or a query related to cars
        query_type_data = detect_query_type(request.question, openai_client, chat_model)
        
        query_type = query_type_data.get("type", "recommendation")
        message = query_type_data.get("message", "")
        
        if query_type == "general":
            log("General question detected, returning response without car search")
            return RagQueryResponse(
                data=message if message else "I'm here to help you find the perfect car from our showroom. Please let me know what car you're looking for!"
            )
        else:
            log(f"Car-related query detected (type: {query_type}), proceeding with car search")

        # Extracting filters from query using GPT (for car-related queries)
        log("Extracting filters from query")
        filters = extract_filters_from_query(request.question, openai_client, chat_model)
        
        # Normalize filters to canonical values (colors and cities)
        log("Normalizing filters to canonical values")
        filters = normalize_filters_to_canonical(filters)
        log(f"Normalized filters: {filters}")
        
        # Calculate dynamic similarity threshold based on filter count
        similarity_threshold = calculate_dynamic_similarity_threshold(
            filters,
            base_threshold=base_similarity_threshold,
            min_threshold=min_similarity_threshold,
            filter_increment=filter_threshold_increment
        )
        log(f"Using dynamic similarity threshold: {similarity_threshold:.2f} (base: {base_similarity_threshold}, min: {min_similarity_threshold})")
        
        log("Model found in filters, using vector similarity search")
        # Creating embedding for the query
        log("Creating embedding for query")
        query_embedding = create_embedding(request.question, openai_client, embedding_model)
        log("Embedding created successfully")
            
        qdrant_filter = build_qdrant_filter(filters)
        query_params = {
            "collection_name": collection_name,
            "query": query_embedding,
            "limit": request.top_k
        }
            
        if qdrant_filter:
            query_params["query_filter"] = qdrant_filter
            log(f"Applying Qdrant filter directly: {qdrant_filter}")
            
        query_result = qdrant_client.query_points(**query_params)
        candidates = query_result.points if hasattr(query_result, 'points') else []
            
        log(f"Vector search returned {len(candidates)} results")
            
        # Filter by dynamic similarity threshold (only for vector search)
        candidates = [
            result for result in candidates
            if result.score >= similarity_threshold
        ]
        log(f"After similarity threshold filtering: {len(candidates)} results")
        
        # Sort by year preference if specified (newest/oldest)
        #candidates = sort_results_by_year_preference(candidates, filters)
        
        # Limit to top_k after sorting
        search_results = candidates[:request.top_k]
        
        log(f"Final results: {len(search_results)} cars")

        # Handle empty results
        if not search_results:
            log("No results found above similarity threshold")
            return RagQueryResponse(
                data="I'm sorry, but I couldn't find any cars that match your criteria. "
                     "Please try adjusting your search terms or consider different specifications."
            )

        # Format car data for prompt (including city and similarity scores)
        log("Formatting car data for prompt")
        cars_list = []
        for idx, doc in enumerate(search_results, 1):
            car = doc.payload if hasattr(doc, 'payload') else doc
            similarity = getattr(doc, 'score', 1.0)  # Default to 1.0 for scroll results
            cars_list.append(
                f"{idx}. Model: {car.get('model', 'N/A')}\n"
                f"   Generation: {car.get('generation', 'N/A')}\n"
                f"   City: {car.get('city', 'N/A')}\n"
                f"   Mileage: {car.get('mileage', 'N/A')}\n"
                f"   Color: {car.get('color', 'N/A')}\n"
                f"   Engine: {car.get('engine', 'N/A')}\n"
                f"   Price: {car.get('price', 'N/A')}\n"
                f"   URL: {car.get('url', 'N/A')}\n"
                f"   Match Score: {similarity:.2%}"
            )

        cars_text = "\n\n".join(cars_list)
        log(f"Formatted {len(cars_list)} cars for prompt")

        log("Sending request to OpenAI chat completion")
        answer = generate_recommendation_response(
            request.question,
            cars_text,
            openai_client,
            chat_model,
            chat_temperature
        )
        log("Successfully generated response")

        return RagQueryResponse(data=answer)

    except Exception as e:
        # Single catch-all exception handler
        log_error(f"Error in search_cars: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )
