# TiddlyWiki AI Integration

Python tools for fetching TiddlyWiki tiddlers, generating embeddings, and storing them in PostgreSQL with pgvector for semantic search.

## Features

- Fetch tiddlers from a TiddlyWiki instance via API
- Generate OpenAI embeddings for tiddler content
- Store tiddlers with embeddings in PostgreSQL using pgvector
- Automatic database schema creation and indexing
- Docker Compose deployment with secrets management

## Installation

### Option 1: Docker Compose (Recommended)

The easiest way to get started is using Docker Compose:

```bash
# 1. Setup secrets
cp secrets/postgres_password.txt.example secrets/postgres_password.txt
cp secrets/openai_api_key.txt.example secrets/openai_api_key.txt
# Edit secret files with your actual values

# 2. Start the database
docker compose up -d

# 3. Use the Python scripts from your host
python tiddlywiki_api.py localhost:8080
```

See [DOCKER.md](DOCKER.md) for complete Docker deployment documentation.

### Option 2: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure PostgreSQL has the pgvector extension installed:
```sql
CREATE EXTENSION vector;
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Environment Variables

Required variables in `.env`:

```
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database Configuration
POSTGRES_HOST=localhost          # Optional, defaults to localhost
POSTGRES_PORT=5432              # Optional, defaults to 5432
POSTGRES_DB=tiddlywiki          # Required
POSTGRES_USER=your_db_user      # Required
POSTGRES_PASSWORD=your_db_pass  # Required
```

## Usage

### Command Line

Fetch tiddlers and save to database:
```bash
python tiddlywiki_api.py localhost:8080
```

Search for similar tiddlers:
```bash
# Search with default top 5 results
python tiddlywiki_api.py search "machine learning concepts"

# Search with custom number of results
python tiddlywiki_api.py search "neural networks" 10
```

### Python API

```python
from tiddlywiki_api import get_tiddlers_with_embeddings, save_tiddlers_to_postgres, search_similar_tiddlers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch tiddlers with embeddings
tiddlers = get_tiddlers_with_embeddings("localhost:8080")

# Save to PostgreSQL
save_tiddlers_to_postgres(tiddlers)

# Search for similar tiddlers
results = search_similar_tiddlers("machine learning", top_k=5)
for result in results:
    print(f"{result['title']}: {result['similarity']:.3f}")
    print(f"  URL: {result['link_url']}")
```

## Database Schema

The function creates a table with the following structure:

```sql
CREATE TABLE tiddlers (
    id SERIAL PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    link_url TEXT NOT NULL,
    download_url TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI embedding dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity search index
CREATE INDEX tiddlers_embedding_idx
ON tiddlers USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## Semantic Search

After storing embeddings, you can perform semantic searches using the built-in function:

```python
from tiddlywiki_api import search_similar_tiddlers
from dotenv import load_dotenv

load_dotenv()

# Search for tiddlers similar to your query
results = search_similar_tiddlers("your search query", top_k=5)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Link: {result['link_url']}\n")
```

The function uses pgvector's cosine similarity to find the most semantically similar tiddlers to your query.

## Functions

### `get_tiddlers(domain: str) -> List[Dict[str, Any]]`
Fetch list of tiddlers from TiddlyWiki instance.

### `get_tiddler_content(domain: str, title: str) -> Dict[str, Any]`
Fetch full content of a single tiddler.

### `get_tiddlers_with_embeddings(domain: str, openai_api_key: str = None) -> List[Dict[str, Any]]`
Fetch all tiddlers and generate embeddings for their content.

### `save_tiddlers_to_postgres(tiddlers: List[Dict[str, Any]], table_name: str = 'tiddlers') -> None`
Save tiddlers with embeddings to PostgreSQL database with pgvector.

### `search_similar_tiddlers(query: str, top_k: int = 5, table_name: str = 'tiddlers', openai_api_key: str = None) -> List[Dict[str, Any]]`
Search for tiddlers most similar to the given query string using semantic search. Returns a list of results with title, URLs, and similarity scores.
