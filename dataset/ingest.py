import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models as rest
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.synonyms import normalize_color_to_canonical, normalize_city_to_canonical

# Load environment variables
load_dotenv()

# Loggers
def log(msg):
    print(f"[INFO] {msg}")

def log_error(msg):
    print(f"[ERROR] {msg}")

# Configuration
COLLECTION_NAME = "cars"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_SIZE = 1536

# Initializing quadrant and openAI clients
log("Initializing quadrant and openAI clients")
qdrant = QdrantClient(host="localhost", port=6333)

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    log_error("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    raise ValueError("OPENAI_API_KEY is required")

client_openai = OpenAI(api_key=openai_api_key)
log("Clients initialized")

# Function which creates semantic description using OpenAI
def create_semantic_description(car):
    """
    Use OpenAI to generate a semantic description of the car for embedding.
    This description focuses on semantic meaning rather than exact filter values.
    """
    prompt = f"""Create a natural, semantic description of this car for search purposes. 
Focus on describing the car in a way that would help someone find it through natural language queries.

Car details:
- Model: {car.get('model', 'N/A')}
- Generation: {car.get('generation', 'N/A')}
- Year: {car.get('modelYear', 'N/A')}
- Color: {car.get('color', 'N/A')}
- Engine: {car.get('engine', 'N/A')}
- Mileage: {car.get('mileage', 'N/A')}
- City: {car.get('city', 'N/A')}
- Price: {car.get('price', 'N/A')}

Write a natural, conversational description that captures the essence of this car.
Focus on semantic meaning - describe what kind of car it is, its characteristics, and what someone might search for.
Do NOT include exact numeric values like specific prices or mileage numbers in the description.
Instead, describe them semantically (e.g., "affordable", "low mileage", "recent model", etc.).

Keep it concise (2-3 sentences) and natural. Write in English."""

    try:
        response = client_openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates natural, semantic descriptions of cars for search purposes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        description = response.choices[0].message.content.strip()
        log(f"Generated semantic description: {description[:100]}...")
        return description
    except Exception as e:
        log_error(f"Error generating semantic description: {e}")
        # Fallback to a simple description if OpenAI fails
        return f"{car.get('model', 'Car')} {car.get('generation', '')} in {car.get('color', '')} color with {car.get('engine', '')} engine"

# Function creating embedings via openAI API
def get_embedding(text):
    response = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def start():
    # Reseting points in collection
    log(f"Resetting collection '{COLLECTION_NAME}'")
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
    log(f"Collection '{COLLECTION_NAME}' reset.")

    # Creating quadrant collection if it doesn't exist
    # using cosine distance (standart for searching semantic close vectors)
    log("Checking existing Qdrant collection")
    collections = qdrant.get_collections().collections
    existing_names = [c.name for c in collections]

    if COLLECTION_NAME not in existing_names:
        log(f"Collection '{COLLECTION_NAME}' not found. Creating...")

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        log(f"Collection '{COLLECTION_NAME}' created.")
    else:
        log(f"Collection '{COLLECTION_NAME}' already exists.")

    # Loading data from JSON-file
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "cars.json")

    if not os.path.exists(file_path):
        log_error(f"File '{file_path}' not found.")
        return
    
    log(f"Loading data from '{file_path}'")
    with open(file_path, "r", encoding="utf-8") as f:
        cars = json.load(f)

    log(f"Loaded {len(cars)} cars")

    log("Creating semantic descriptions and embeddings, then saving to Qdrant")
    points = []
    for i, car in enumerate(cars):
        log(f"Processing car {i+1}/{len(cars)}: {car.get('model', 'N/A')}")
        
        # Normalize color and city to canonical values
        car_normalized = car.copy()
        
        # Normalize color: save raw color and add canonical color
        original_color = car.get('color', '')
        if original_color:
            car_normalized['color_raw'] = original_color
            canonical_color = normalize_color_to_canonical(original_color)
            if canonical_color:
                car_normalized['color'] = canonical_color
                if original_color.lower() != canonical_color.lower():
                    log(f"  Color normalized: '{original_color}' → '{canonical_color}'")
            else:
                # If normalization fails, keep original as canonical
                log(f"  Color normalization failed for '{original_color}', keeping as-is")
        
        # Normalize city: save raw city and add canonical city
        original_city = car.get('city', '')
        if original_city:
            car_normalized['city_raw'] = original_city
            canonical_city = normalize_city_to_canonical(original_city)
            if canonical_city:
                car_normalized['city'] = canonical_city
                if original_city.lower() != canonical_city.lower():
                    log(f"  City normalized: '{original_city}' → '{canonical_city}'")
            else:
                # If normalization fails, keep original as canonical
                log(f"  City normalization failed for '{original_city}', keeping as-is")
        
        # Generate semantic description using OpenAI (use original values for description)
        semantic_description = create_semantic_description(car)
        
        # Create embedding from semantic description
        embedding = get_embedding(semantic_description)
        
        # Save embedding with normalized car data in payload
        points.append({
            "id": i,
            "vector": embedding,
            "payload": car_normalized  # Normalized data with color_raw and city_raw
        })
        
        # Log progress every 10 cars
        if (i + 1) % 10 == 0:
            log(f"Processed {i + 1}/{len(cars)} cars...")
    
    log(f"Saving {len(points)} points to Qdrant")
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    log(f"Points saved to Qdrant")
    log("Ingestion completed successfully")


if __name__ == "__main__":
    start()