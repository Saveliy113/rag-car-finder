# RAG Car Finder

A Retrieval-Augmented Generation (RAG) system for intelligent car search and recommendation, specifically designed for Toyota Camry vehicles. This system combines semantic vector search with structured filtering to provide natural language-based car discovery and personalized recommendations.

## Main Idea

The main idea of this test project is to allow users to find cars using natural language queries. Instead of traditional keyword-based search, users can ask questions like "Find me a silver Toyota Camry under 5 million tenge in Almaty" and receive intelligent, context-aware recommendations.

The system leverages:
- **Semantic Understanding**: Uses OpenAI embeddings to understand the meaning behind user queries
- **Structured Filtering**: Extracts precise filters (price, mileage, color, city, etc.) from natural language
- **Hybrid Search Strategy**: Combines vector similarity search with exact filtering for optimal results
- **Intelligent Recommendations**: Uses GPT to generate natural, conversational responses with personalized car suggestions

## Concepts

### Retrieval-Augmented Generation (RAG)

The system follows the RAG pattern:
1. **Retrieval**: Searches a vector database (Qdrant) to find relevant cars based on semantic similarity
2. **Augmentation**: Enriches the search with structured filters extracted from natural language
3. **Generation**: Uses GPT to synthesize the retrieved information into a helpful, conversational response

### Dual Search Strategy

The system employs two search modes:

1. **Vector Similarity Search** (when car model is mentioned):
   - Creates embeddings for user queries
   - Performs semantic similarity search in Qdrant
   - Applies structured filters as post-processing
   - Filters results by similarity threshold (≥0.5)

2. **Filtered Scroll Search** (when no model specified):
   - Retrieves all cars matching extracted filters
   - Uses Qdrant's scroll API for efficient pagination
   - Applies all filters at the database level

### Filter Extraction

Uses GPT-4o-mini to extract structured filters from natural language:
- **Price**: min/max price in tenge
- **Mileage**: min/max mileage in kilometers
- **Model**: Car model name (Toyota Camry, Camry, etc.)
- **Color**: Car color (exact match)
- **City**: Location (exact match)
- **Year**: Specific year or preference (newest/oldest)
- **Engine**: Engine type and fuel

## Design Details

### Architecture

```
┌─────────────┐
│   Client    │ (Frontend on Vue 3 on localhost:9000)
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────────────────┐
│      FastAPI Application         │
│  ┌───────────────────────────┐  │
│  │   RAG Endpoint (/rag/search)│  │
│  └───────────┬────────────────┘  │
│              │                    │
│  ┌───────────▼────────────────┐  │
│  │  Filter Extraction (GPT)    │  │
│  └───────────┬────────────────┘  │
│              │                    │
│  ┌───────────▼────────────────┐  │
│  │  Embedding Generation      │  │
│  │  (OpenAI Embeddings)       │  │
│  └───────────┬────────────────┘  │
│              │                    │
│  ┌───────────▼────────────────┐  │
│  │  Vector Search / Filter    │  │
│  │  (Qdrant Client)           │  │
│  └───────────┬────────────────┘  │
│              │                    │
│  ┌───────────▼────────────────┐  │
│  │  Response Generation (GPT) │  │
│  └────────────────────────────┘  │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│      Qdrant Vector DB           │
│  ┌───────────────────────────┐  │
│  │  Collection: "cars"       │  │
│  │  Vector Size: 1536        │  │
│  │  Distance: Cosine         │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

### Components

#### 1. **API Layer** (`endpoints/`)
- **`rag.py`**: Main RAG search endpoint with filter extraction and response generation

#### 4. **Data Ingestion** (`dataset/`)
- **`ingest.py`**: Script to process JSON data, generate embeddings, and populate Qdrant
- **`cars.json`**: Source dataset with car listings

### Request Flow

1. **User Query**: Client sends natural language query to `/rag/search`
2. **Filter Extraction**: GPT extracts structured filters from query
3. **Search Strategy Selection**:
   - If model mentioned → Vector similarity search with filters
   - If no model → Filtered scroll search
4. **Result Processing**:
   - Apply similarity threshold (vector search only)
   - Sort by year preference if specified
   - Limit to top_k results
5. **Response Generation**: GPT generates natural language recommendation
6. **Response**: Returns formatted JSON with recommendation text

### Configuration

Key configuration constants in `endpoints/rag.py`:
- `SIMILARITY_THRESHOLD = 0.5`: Minimum similarity score for vector search results
- `EMBEDDING_MODEL = "text-embedding-3-small"`: OpenAI embedding model (1536 dimensions)
- `CHAT_MODEL = "gpt-4o-mini"`: GPT model for filter extraction and response generation
- `CHAT_TEMPERATURE = 0.7`: Temperature for response generation (0.1 for filter extraction)
- `COLLECTION_NAME = "cars"`: Qdrant collection name

## Dataset Concept

### Data Structure

The dataset (`dataset/cars.json`) contains car listings in JSON format with the following fields:

```json
{
  "city": "Алматы",
  "model": "Toyota Camry 2005 г.",
  "generation": "2004 - 2006 XV30 рестайлинг (V35)",
  "mileage": "286 000 км",
  "color": "серебристый металлик",
  "engine": "2.4 (бензин)",
  "price": "4 800 000 ₸",
  "url": "https://kolesa.kz/a/show/166222492",
  "modelYear": 2005,
  "price_num": 4800000,
  "mileage_num": 286000
}
```

### Embedding Strategy

For vector search, each car is embedded using a composite text representation:

```
"{model}, {generation}, {mileage}, {color}, {engine}, цена {price}"
```

This approach ensures that:
- Semantic similarity captures car characteristics
- Model and generation information is prominent
- Price and mileage are included for relevance
- The embedding reflects the full context of the listing

### Data Processing

The ingestion script (`dataset/ingest.py`):
1. Loads car data from JSON file
2. Creates or resets Qdrant collection
3. Generates embeddings for each car using OpenAI API
4. Stores vectors with full car metadata as payload
5. Uses numeric fields (`price_num`, `mileage_num`, `modelYear`) for efficient filtering

## System Technical Details

### Technology Stack

- **Framework**: FastAPI (Python web framework)
- **Vector Database**: Qdrant (open-source vector similarity search engine)
- **AI/ML**:
  - OpenAI GPT-4o-mini (filter extraction and response generation)
  - OpenAI text-embedding-3-small (semantic embeddings)

### API Endpoints

#### `POST /rag/search`

Search for cars using natural language query.

**Request Body**:
```json
{
  "question": "Find me a silver Toyota Camry under 5 million tenge in Almaty",
  "top_k": 5
}
```

**Response**:
```json
{
  "data": "Based on your requirements, I found several excellent options..."
}
```

**Parameters**:
- `question` (string, required): Natural language search query
- `top_k` (integer, 1-20, default: 5): Number of results to return

## Requirements

For the most precise search results, it is recommended to use **exact model and filter values** that match the dataset format:

- **Model**: Use exact model names as they appear in the dataset (e.g., "Toyota Camry 2005 г.") rather than partial matches (e.g., "Camry")
- **Color**: Use exact color values from the dataset (e.g., "серебристый металлик") rather than synonyms or translations
- **City**: Use exact city names as stored (e.g., "Алматы")
- **Engine**: Use exact engine format (e.g., "2.4 (бензин)")
- **Price/Mileage**: Use numeric values in the same units (tenge for price, kilometers for mileage)

While the system can handle natural language queries and will attempt to extract filters automatically, using exact values ensures:
- Better filter matching (exact match filters work more reliably)
- More accurate results
- Faster query processing
- Reduced ambiguity in filter extraction

## Limitations

### Current Constraints

1. **Single Car Model**: The system is currently specialized for Toyota Camry vehicles only. Queries for other car models will be politely declined.

2. **Exact Match Filters**: Some filters (color, city, engine) use exact matching, which may miss variations or synonyms (e.g., "silver" vs "серебристый").

3. **Similarity Threshold**: Vector search results are filtered by a fixed similarity threshold (0.5), which may exclude relevant results in some cases.

5. **Model Filter**: The model filter uses exact matching, which may not handle partial matches well (e.g., "Camry" vs "Toyota Camry 2005 г.").
