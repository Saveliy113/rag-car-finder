import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models as rest

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

# Function which creates embedding from key words
def create_text_for_embedding(car):
    return f"{car['model']}, {car['generation']}, {car['mileage']}, {car['color']}, {car['engine']}, цена {car['price']}"

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

    log("Creating embeddings and saving to Qdrant")
    points = []
    for i, car in enumerate(cars):
        log(f"Car: {car}")
        text = create_text_for_embedding(car)
        embedding = get_embedding(text)
        points.append({
            "id": i,
            "vector": embedding,
            "payload": car # car metadata
        })
    
    log(f"Saving {len(points)} points to Qdrant")
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    log(f"Points saved to Qdrant")
    log("Ingestion completed successfully")


if __name__ == "__main__":
    start()