from fastapi import APIRouter, HTTPException
from openai import OpenAIError, APIError
from qdrant_client.http.exceptions import UnexpectedResponse

from models.models import RagQueryRequest
from loaders import get_qdrant_client, get_openai_client

# Loggers
def log(msg):
    print(f"[INFO] {msg}")

def log_error(msg):
    print(f"[ERROR] {msg}")

router = APIRouter(prefix="/rag", tags=["RAG"])

# Configuration constants
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score to include results
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHAT_TEMPERATURE = 0.7
COLLECTION_NAME = "cars"


@router.post("/search")
async def search_cars(request: RagQueryRequest):
    """
    Search for cars using RAG (Retrieval-Augmented Generation)

    This endpoint:
    1. Creates an embedding for the user's query
    2. Searches the vector database for similar cars
    3. Uses GPT to generate a natural language response with recommendations
    """
    try:
        # Get clients (already initialized at startup)
        qdrant_client = get_qdrant_client()
        openai_client = get_openai_client()

        log(f"Processing query: {request.question}")

        # Step 1: Create embedding for the query
        try:
            log("Creating embedding for query")
            embedding_response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=request.question
            )
            query_embedding = embedding_response.data[0].embedding
            log("Embedding created successfully")
        except (OpenAIError, APIError) as e:
            log_error(f"OpenAI embedding error: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to create embedding: {str(e)}"
            )

        # Step 2: Search in Qdrant
        try:
            log(f"Searching Qdrant with top_k={request.top_k}")
            
            # Use query_points - the correct method for Qdrant client
            # It returns a QueryResult object with .points attribute
            query_result = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=request.top_k
            )
            
            # Extract points from QueryResult - it has a .points attribute
            search_results = query_result.points

            # Filter results by similarity threshold
            filtered_results = [
                result for result in search_results
                if result.score >= SIMILARITY_THRESHOLD
            ]
            log(
                f"Found {len(filtered_results)} results above threshold "
                f"{SIMILARITY_THRESHOLD} (from {len(search_results)} total)"
            )
            search_results = filtered_results
        except (UnexpectedResponse, Exception) as e:
            log_error(f"Qdrant search error: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Vector database search failed: {str(e)}"
            )

        # Step 3: Handle empty results
        if not search_results:
            log("No results found above similarity threshold")
            return (
                "I'm sorry, but I couldn't find any Toyota Camry cars that "
                "match your criteria. Please try adjusting your search terms "
                "or consider different specifications."
            )

        # Step 4: Format car data for prompt (including city and similarity scores)
        log("Formatting car data for prompt")
        cars_list = []
        for idx, doc in enumerate(search_results, 1):
            car = doc.payload
            similarity = doc.score
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

        # Step 5: Create structured prompt
        system_message = (
            "You are an expert car consultant specializing in Toyota Camry "
            "vehicles. Your role is to help customers find the best car that "
            "matches their needs and preferences. You provide clear, helpful "
            "recommendations based on the available inventory."
        )

        user_prompt = f"""A customer is asking: "{request.question}"

Below are Toyota Camry cars from our inventory that match their query (sorted by relevance):

{cars_text}

Please provide a helpful recommendation following these guidelines:
1. Analyze the customer's question and identify their key requirements (price range, mileage, color, location, etc.)
2. Recommend the best matching car(s) from the list above (maximum 3 cars)
3. For each recommended car, clearly state why it matches their needs
4. If the customer asks for a car model other than Toyota Camry, politely explain that we only have Toyota Camry vehicles available
5. Format your response in a friendly, conversational manner
6. Include the URL for each recommended car so the customer can view more details

If no cars truly match the customer's requirements, politely explain this and suggest alternative criteria."""

        log("Sending request to OpenAI chat completion")

        # Step 6: Generate response using OpenAI
        try:
            chat_response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=CHAT_TEMPERATURE,
                max_tokens=1000
            )

            answer = chat_response.choices[0].message.content
            log("Successfully generated response")

            return answer

        except (OpenAIError, APIError) as e:
            log_error(f"OpenAI chat completion error: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to generate response: {str(e)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Handle validation errors
        log_error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch-all for unexpected errors
        log_error(f"Unexpected error in search_cars: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
