import json
import re
from fastapi import APIRouter, HTTPException
from openai import OpenAIError, APIError
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

from models.models import RagQueryRequest, RagQueryResponse
from loaders import get_qdrant_client, get_openai_client

# Loggers
def log(msg):
    print(f"[INFO] {msg}")

def log_error(msg):
    print(f"[ERROR] {msg}")

router = APIRouter(prefix="/rag", tags=["RAG"])

# Configuration constants
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score to include results
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHAT_TEMPERATURE = 0.7
COLLECTION_NAME = "cars"


def extract_filters_from_query(query: str, openai_client) -> dict:
    """
    Use GPT to extract structured filters from natural language query
    Returns a dictionary with extracted filters
    """
    filter_prompt = f"""Extract filters from this car search query in JSON format.
Query: "{query}"

Extract the following information if mentioned:
- model: Car model name 
- max_price: Maximum price in tenge (extract numeric value, e.g., "до 15 000 000 тенге" -> 15000000)
- min_price: Minimum price in tenge
- max_mileage: Maximum mileage in km (extract numeric value)
- min_mileage: Minimum mileage in km
- color: Car color (exact match)
- city: City name (exact match)
- year_preference: "newest", "oldest", or specific year (e.g., 2020)
- engine: Engine type (e.g., "2.5 (бензин)")

Return ONLY valid JSON in this format:
{{
    "model": null or string,
    "max_price": null or number,
    "min_price": null or number,
    "max_mileage": null or number,
    "min_mileage": null or number,
    "color": null or string,
    "city": null or string,
    "year_preference": null or "newest" or "oldest" or year number,
    "engine": null or string
}}

If a filter is not mentioned, use null. Return ONLY the JSON, no other text."""

    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from queries. Return only valid JSON."},
                {"role": "user", "content": filter_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=200
        )
        
        json_str = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        
        filters = json.loads(json_str)
        log(f"Extracted filters: {filters}")
        return filters
    except Exception as e:
        log_error(f"Error extracting filters: {e}")
        return {}


def build_qdrant_filter(filters: dict) -> Filter:
    """
    Build Qdrant Filter object from extracted filters
    Uses numeric fields: price_num, mileage_num, year_num
    """
    conditions = []
    
    # Price filter (using price_num field)
    if filters.get("max_price") is not None and filters.get("min_price") is not None:
        # Both min and max price
        conditions.append(
            FieldCondition(
                key="price_num",
                range=Range(gte=filters["min_price"], lte=filters["max_price"])
            )
        )
    elif filters.get("max_price") is not None:
        conditions.append(
            FieldCondition(
                key="price_num",
                range=Range(lte=filters["max_price"])
            )
        )
    elif filters.get("min_price") is not None:
        conditions.append(
            FieldCondition(
                key="price_num",
                range=Range(gte=filters["min_price"])
            )
        )
    
    # Mileage filter (using mileage_num field)
    if filters.get("max_mileage") is not None and filters.get("min_mileage") is not None:
        # Both min and max mileage
        conditions.append(
            FieldCondition(
                key="mileage_num",
                range=Range(gte=filters["min_mileage"], lte=filters["max_mileage"])
            )
        )
    elif filters.get("max_mileage") is not None:
        conditions.append(
            FieldCondition(
                key="mileage_num",
                range=Range(lte=filters["max_mileage"])
            )
        )
    elif filters.get("min_mileage") is not None:
        conditions.append(
            FieldCondition(
                key="mileage_num",
                range=Range(gte=filters["min_mileage"])
            )
        )
    
    # Year filter (using modelYear field)
    if filters.get("year_preference") and isinstance(filters["year_preference"], int):
        # Specific year
        conditions.append(
            FieldCondition(
                key="modelYear",
                match=MatchValue(value=filters["year_preference"])
            )
        )
        # "newest" and "oldest" will be handled by sorting after search
    
    # Color filter
    if filters.get("color"):
        conditions.append(
            FieldCondition(
                key="color",
                match=MatchValue(value=filters["color"])
            )
        )
    
    # City filter
    if filters.get("city"):
        conditions.append(
            FieldCondition(
                key="city",
                match=MatchValue(value=filters["city"])
            )
        )
    
    # Engine filter
    if filters.get("engine"):
        conditions.append(
            FieldCondition(
                key="engine",
                match=MatchValue(value=filters["engine"])
            )
        )
    
    if conditions:
        return Filter(must=conditions)
    return None

def sort_results_by_year_preference(results, filters: dict):
    """Sort results by year preference (newest/oldest) if specified"""
    if not filters.get("year_preference"):
        return results
    
    if filters["year_preference"] == "newest":
        # Sort by modelYear descending
        return sorted(
            results,
            key=lambda x: x.payload.get("modelYear") or 0,
            reverse=True
        )
    elif filters["year_preference"] == "oldest":
        # Sort by modelYear ascending
        return sorted(
            results,
            key=lambda x: x.payload.get("modelYear") or 9999
        )
    
    return results


@router.post("/search", response_model=RagQueryResponse)
async def search_cars(request: RagQueryRequest):
    """
    Search for cars using RAG (Retrieval-Augmented Generation)

    This endpoint:
    1. Preprocesses query with GPT to extract filters
    2. Creates an embedding for the user's query
    3. Searches the vector database with filters
    4. Uses GPT to generate a natural language response with recommendations
    """
    try:
        # Get clients
        qdrant_client = get_qdrant_client()
        openai_client = get_openai_client()

        log(f"Processing query: {request.question}")

        # Extracting filters from query using GPT
        log("Extracting filters from query")
        filters = extract_filters_from_query(request.question, openai_client)
        
        model_in_query = filters.get("model")
        # If model is specified - do embedding + query_points
        if model_in_query:
            log("Model found in filters, using vector similarity search")
            # Creating embedding for the query
            log("Creating embedding for query")
            embedding_response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=request.question
            )
            query_embedding = embedding_response.data[0].embedding
            log("Embedding created successfully")
            
            qdrant_filter = build_qdrant_filter(filters)
            query_params = {
                "collection_name": COLLECTION_NAME,
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
                if result.score >= SIMILARITY_THRESHOLD
            ]
        else:
            # If model is NOT specified - generate polite response asking for clarification
            log("No model specified in query, generating polite response to ask for details")
            
            clarification_prompt = """A customer has contacted our car selling service but hasn't specified which car model they're looking for.

Generate a polite, friendly, and welcoming response in English asking them to specify:
- The exact car model they're interested in
- Their preferences (price range, mileage, color, city, engine type, year, etc.)

IMPORTANT RULES:
- DO NOT use placeholders, brackets, variables, or any template-like content (e.g., [Customer's Name], {{name}}, etc.)
- DO NOT include any information that wasn't provided by the customer
- Write a direct, concrete response as if you're speaking directly to the customer
- Keep the tone professional, helpful, and welcoming
- Make it clear that providing more details will help us find the best car for their needs
- Write 2-3 sentences maximum, be concise and direct

Example of what to write (use this style but make it natural):
"Hello and welcome to our car selling service. To find the best car for your needs, we need you to specify the exact model and your preferences. Please let us know what car model you're interested in, along with any preferences regarding price, mileage, color, city, engine type, or year. We're waiting for your information!"

Example of what NOT to write (avoid this):
"Dear [Customer's Name], thank you for contacting us... [insert details here]..." """

            try:
                clarification_response = openai_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a friendly and helpful car sales assistant. Always be polite and welcoming. Respond in English. Never use placeholders, brackets, or template variables. Always write direct, concrete responses."},
                        {"role": "user", "content": clarification_prompt}
                    ],
                    temperature=0.5,
                    max_tokens=200
                )
                
                polite_answer = clarification_response.choices[0].message.content.strip()
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

        # Create structured prompt
        system_message = (
            "You are an expert car consultant. Your role is to help customers find the best car that "
            "matches their needs and preferences. You provide clear, helpful "
            "recommendations based on the available inventory."
        )

        user_prompt = f"""A customer is asking: "{request.question}"

Below are cars from our inventory that match their query (sorted by relevance):

{cars_text}

Please provide a helpful recommendation following these guidelines:
1. Analyze the customer's question and identify their key requirements (price range, mileage, color, location, etc.)
2. Recommend the best matching car(s) from the list above
3. For each recommended car, clearly state why it matches their needs
4. Format your response in a friendly, conversational manner
5. Include the URL for each recommended car so the customer can view more details

If no cars truly match the customer's requirements, politely explain this and suggest alternative criteria."""

        log("Sending request to OpenAI chat completion")

        # Generate response using OpenAI
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

        return RagQueryResponse(data=answer)

    except Exception as e:
        # Single catch-all exception handler
        log_error(f"Error in search_cars: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )
