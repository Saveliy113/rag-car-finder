"""
OpenAI query functions for RAG car finder.
Handles all interactions with OpenAI API including prompts and responses.
"""
import json
import re
from typing import Dict, Any

from utils.logger import log, log_error


def get_filter_extraction_prompt(query: str) -> str:
    """Generate prompt for extracting filters from user query"""
    return f"""Extract filters from this car search query in JSON format.
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


def extract_filters_from_query(query: str, openai_client, chat_model: str) -> Dict[str, Any]:
    """
    Use GPT to extract structured filters from natural language query
    Returns a dictionary with extracted filters
    """
    filter_prompt = get_filter_extraction_prompt(query)

    try:
        response = openai_client.chat.completions.create(
            model=chat_model,
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


def create_embedding(query: str, openai_client, embedding_model: str) -> list:
    """
    Create embedding vector for a query
    Returns the embedding vector
    """
    try:
        embedding_response = openai_client.embeddings.create(
            model=embedding_model,
            input=query
        )
        return embedding_response.data[0].embedding
    except Exception as e:
        log_error(f"Error creating embedding: {e}")
        raise
        
def get_recommendation_system_message() -> str:
    """Get system message for car recommendation"""
    return (
        "You are an expert car consultant. Your role is to help customers find the best car that "
        "matches their needs and preferences. You provide clear, helpful "
        "recommendations based on the available inventory."
    )


def get_recommendation_user_prompt(query: str, cars_text: str) -> str:
    """Generate user prompt for car recommendation"""
    return f"""A customer is asking: "{query}"

Below are cars from our inventory that match their query (sorted by relevance):

{cars_text}

Please provide a helpful recommendation following these guidelines:
1. Analyze the customer's question and identify their key requirements (price range, mileage, color, location, etc.)
2. Recommend the best matching car(s) from the list above
3. For each recommended car, clearly state why it matches their needs
4. Format your response in a friendly, conversational manner
5. Include the URL for each recommended car so the customer can view more details

If no cars truly match the customer's requirements, politely explain this and suggest alternative criteria."""


def generate_recommendation_response(
    query: str,
    cars_text: str,
    openai_client,
    chat_model: str,
    chat_temperature: float
) -> str:
    """
    Generate a natural language recommendation response based on search results
    Returns the generated response text
    """
    system_message = get_recommendation_system_message()
    user_prompt = get_recommendation_user_prompt(query, cars_text)
    
    try:
        chat_response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=chat_temperature,
            max_tokens=1000
        )
        
        return chat_response.choices[0].message.content
    except Exception as e:
        log_error(f"Error generating recommendation response: {e}")
        raise


def detect_query_type(query: str, openai_client, chat_model: str) -> dict:
    """
    Detect the type of query - general question or a query related to cars
    Returns a dictionary with 'type' and 'message' keys
    """
    try:
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": """
                You are an expert car consultant. Your task is to meet clients in our system and help them find the best car that matches their needs and preferences.
                You needto detect if the client is asking a general question or a query related to cars.
                The outpust format should be a JSON object with the following fields:
                - type: "general" or "recommendation" - boolean value
                - message: Leave empty if type is "recommendation" or a generated response if type is "general"
                For any general question you should generate a polite and helpful response that will help the client to understand that we are responsible only for choosing cars in our showroom.
                Don't provide any information which is not related to our services.
                """},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        # Parse the JSON response
        json_str = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        query_type_data = json.loads(json_str)
        
        return query_type_data
    except json.JSONDecodeError as e:
        log_error(f"Error parsing query type JSON: {e}")
        return {"type": "recommendation", "message": ""}
    except Exception as e:
        log_error(f"Error detecting query type: {e}")
        return {"type": "recommendation", "message": ""}