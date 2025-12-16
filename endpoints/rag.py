import os
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from utils.logger import log, log_error
from utils.openai_queries import (
    extract_filters_from_query,
    create_embedding,
    generate_clarification_response,
    generate_recommendation_response
)
from utils.rag_filters import build_qdrant_filter, sort_results_by_year_preference
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
        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        chat_temperature = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
        collection_name = os.getenv("COLLECTION_NAME", "cars")

        log(f"Processing query: {request.question}")

        # Extracting filters from query using GPT
        log("Extracting filters from query")
        filters = extract_filters_from_query(request.question, openai_client, chat_model)
        
        model_in_query = filters.get("model")
        # If model is specified - do embedding + query_points
        if model_in_query:
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
            
            # Filter by similarity threshold (only for vector search)
            candidates = [
                result for result in candidates
                if result.score >= similarity_threshold
            ]
        else:
            # If model is NOT specified - generate polite response asking for clarification
            log("No model specified in query, generating polite response to ask for details")
            
            try:
                polite_answer = generate_clarification_response(openai_client, chat_model)
                log(f"Generated clarification response: {polite_answer[:100]}...")
                return RagQueryResponse(data=polite_answer)
            except Exception as e:
                log_error(f"Error generating clarification response: {e}")
                # Fallback response if LLM fails
                return RagQueryResponse(
                    data="Hello and welcome to our car selling service! To find the best car for your needs, "
                         "we need you to specify the exact model and your preferences (price, mileage, color, "
                         "city, engine type, etc.). We're waiting for information from you!"
                )
        
        # Sort by year preference if specified (newest/oldest)
        candidates = sort_results_by_year_preference(candidates, filters)
        
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
