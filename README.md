# TiddlyWiki AI Integration

Python tools for fetching TiddlyWiki tiddlers, generating embeddings, and storing them in PostgreSQL with pgvector for semantic search and question answering.

## Features

- **Fetch tiddlers** from TiddlyWiki instances via API with optional filtering
- **Generate OpenAI embeddings** for tiddler content with automatic HTML/wikitext stripping
- **Hybrid search** combining exact match, full-text search, and semantic similarity
- **Question answering** using RAG (Retrieval-Augmented Generation) with LangChain
- **Text processing utilities** for HTML stripping, wikitext removal, and heading-based splitting
- **PostgreSQL storage** with pgvector for efficient vector similarity search
- **Automatic database** schema creation and indexing
- **Docker Compose** deployment with secrets management

## Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd "tw-ai"
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install TiddlyWiki (if not already installed)**
```bash
npm install -g tiddlywiki
```

5. **Set up PostgreSQL with pgvector**

   **Option A: Docker Compose (Recommended)**
   ```bash
   # Start PostgreSQL with pgvector
   docker compose up -d
   ```

   **Option B: Local PostgreSQL**
   ```bash
   # Install pgvector extension in your PostgreSQL database
   psql -d your_database -c "CREATE EXTENSION vector;"
   ```

6. **Start local TiddlyWiki servers (automatically downloads latest content)**
```bash
# Download latest content from websites and start servers
python tiddlywiki_api.py serve
```

This command will:
- Delete existing TW.org, GrokTW, and DevTW folders (ensuring fresh content)
- Download the latest HTML from each website
- Import the HTML into each TiddlyWiki folder
- Copy contents from "TW Hidden Settings" folder into each wiki's /tiddlers subfolder
- Start local servers on ports 8081, 8082, and 8083

This ensures you always have the most up-to-date content with custom settings before indexing.

7. **Index your TiddlyWikis** (in a new terminal)
```bash
# Reindex all configured TiddlyWiki instances
python tiddlywiki_api.py reindex
```

8. **Search and ask questions**
```bash
# Search for tiddlers
python tiddlywiki_api.py search "your search query"

# Ask questions
python tiddlywiki_api.py ask "How do I create a custom widget?"
```

See [DOCKER.md](DOCKER.md) for complete Docker deployment documentation.

## Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database Configuration
POSTGRES_HOST=localhost              # Optional, defaults to localhost
POSTGRES_PORT=5432                   # Optional, defaults to 5432
POSTGRES_DB=tiddlywiki_rag          # Required
POSTGRES_USER=your_db_user           # Required
POSTGRES_PASSWORD=your_db_password   # Required
```

## Usage

### Command Line Interface

**Start local TiddlyWiki servers:**
```bash
# Download latest content and start all servers
python tiddlywiki_api.py serve
```

This command:
1. **Deletes** existing folders (TW.org, GrokTW, DevTW) to ensure fresh content
2. **Downloads** the latest HTML from each website
3. **Imports** the content into TiddlyWiki folders
4. **Copies** custom settings from "TW Hidden Settings" folder into each wiki
5. **Starts** servers on ports 8081, 8082, and 8083

This ensures you always have the most recent content from:
- `TW.org` (port 8081) - https://tiddlywiki.com
- `GrokTW` (port 8082) - https://groktiddlywiki.com
- `DevTW` (port 8083) - https://tiddlywiki.com/dev

Press `Ctrl+C` to stop all servers.

**Search for tiddlers (hybrid search):**
```bash
# Default: returns top 5 results
python tiddlywiki_api.py search "your search query"

# Custom number of results
python tiddlywiki_api.py search "machine learning" 10
```

The search uses hybrid ranking that combines:
- Exact keyword matching (ILIKE)
- Full-text search (PostgreSQL tsquery)
- Semantic similarity (vector embeddings)

**Ask questions with RAG:**
```bash
# Default: uses top 5 tiddlers with gpt-4o-mini
python tiddlywiki_api.py ask "How do I create a custom widget?"

# Custom parameters
python tiddlywiki_api.py ask "What are filters?" 10 gpt-4o
```

**Reindex all TiddlyWiki instances:**
```bash
# Deletes existing data and reindexes from all configured local instances
python tiddlywiki_api.py reindex
```

This command:
1. Deletes all existing embeddings (with confirmation)
2. Connects to all three local TiddlyWiki servers (ports 8081, 8082, 8083)
3. Fetches tiddlers from each instance
4. Generates embeddings for all content
5. Saves everything to PostgreSQL

**Note:** Ensure the TiddlyWiki servers are running (`serve` command) before running `reindex`.

**Delete all embeddings:**
```bash
python tiddlywiki_api.py delete
```

### Python API

**Basic workflow:**
```python
from tiddlywiki_api import (
    get_tiddlers_with_embeddings,
    save_tiddlers_to_postgres,
    search_similar_tiddlers_with_text,
    answer_question_with_tiddlers
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch tiddlers with embeddings
tiddlers = get_tiddlers_with_embeddings(
    scan_domain="localhost:8080",
    link_domain="www.example.com"
)

# Save to PostgreSQL
save_tiddlers_to_postgres(tiddlers)

# Hybrid search (exact + full-text + semantic)
results = search_similar_tiddlers_with_text("machine learning", top_k=5)
for result in results:
    print(f"{result['title']}: {result['rank']:.3f}")
    print(f"  URL: {result['link_url']}")
    print(f"  Match type: {result['match_type']}")

# Ask questions with RAG
answer = answer_question_with_tiddlers(
    question="How do I create a custom widget?",
    top_k=5,
    model="gpt-4o-mini"
)
print(answer['answer'])
for source in answer['sources']:
    print(f"  - {source['title']} (rank: {source['rank']:.3f})")
```

**Text processing utilities:**
```python
from tiddlywiki_api import strip_html, strip_wikitext, split_by_headings

# Strip HTML tags and entities
clean_text = strip_html("<p>Hello &amp; <strong>welcome</strong>!</p>")
# Returns: "Hello & welcome!"

# Remove TiddlyWiki wikitext markup
plain_text = strip_wikitext("''bold'' and //italic// with [[links]]")
# Returns: "bold and italic with links"

# Split content by headings
sections = split_by_headings("""
! Main Heading
Content for main section

!! Subsection
Content for subsection
""")
# Returns: [
#   {'! Main Heading': '! Main Heading\nContent for main section\n'},
#   {'!! Subsection': '!! Subsection\nContent for subsection\n'}
# ]
```

## Database Schema

The system automatically creates a table with the following structure:

```sql
CREATE TABLE tiddlers (
    id SERIAL PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    link_url TEXT NOT NULL,          -- URL for viewing in browser
    download_url TEXT NOT NULL,       -- API endpoint URL
    embedding vector(1536),           -- OpenAI embedding (text-embedding-3-small)
    text TEXT NOT NULL,               -- Full text content (stripped of HTML/wikitext)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity search index (IVFFlat for efficient approximate nearest neighbor)
CREATE INDEX tiddlers_embedding_idx
ON tiddlers USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## Search Methods

The system provides three complementary search methods that are combined in hybrid search:

### 1. Exact Match Search
Fast keyword matching using PostgreSQL `ILIKE`:
```python
from tiddlywiki_api import exact_search
results = exact_search("TiddlyWiki", top_k=5)
```

### 2. Full-Text Search
PostgreSQL full-text search with ranking:
```python
from tiddlywiki_api import full_text_search
results = full_text_search("custom widget procedure", top_k=5)
```

### 3. Semantic Similarity Search
Vector similarity using pgvector cosine distance:
```python
from tiddlywiki_api import similarity_search
results = similarity_search("how to customize appearance", top_k=5)
```

### 4. Hybrid Search (Recommended)
Combines all three methods with automatic deduplication and ranking:
```python
from tiddlywiki_api import search_similar_tiddlers_with_text
results = search_similar_tiddlers_with_text("custom styling", top_k=5)
```

## API Reference

### Core Functions

#### `get_tiddlers(domain: str, filter: str = None) -> List[Dict[str, Any]]`
Fetch list of tiddlers from a TiddlyWiki instance via API.
- `domain`: TiddlyWiki domain (e.g., "localhost:8080")
- `filter`: Optional TiddlyWiki filter expression

#### `get_tiddler_content(domain: str, title: str) -> Dict[str, Any]`
Fetch the full content of a single tiddler including its text.

#### `get_tiddlers_with_embeddings(scan_domain: str, link_domain: str, filter: str = None, openai_api_key: str = None) -> List[Dict[str, Any]]`
Fetch all tiddlers, strip HTML/wikitext, and generate OpenAI embeddings.
- `scan_domain`: Domain to fetch tiddlers from
- `link_domain`: Domain to use in generated URLs (for results display)
- `filter`: Optional TiddlyWiki filter expression
- Returns: List of dicts with `title`, `link_url`, `download_url`, `embedding`, `text`

#### `save_tiddlers_to_postgres(tiddlers: List[Dict[str, Any]], table_name: str = 'tiddlers') -> None`
Save tiddlers with embeddings to PostgreSQL. Creates table and indexes automatically.

### Search Functions

#### `exact_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]`
Fast exact keyword matching using PostgreSQL `ILIKE`.

#### `full_text_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]`
Full-text search using PostgreSQL `tsquery` with ranking.

#### `similarity_search(query: str, top_k: int = 5, openai_api_key: str = None) -> List[Dict[str, Any]]`
Semantic similarity search using vector embeddings and cosine distance.

#### `search_similar_tiddlers_with_text(query: str, top_k: int = 5, openai_api_key: str = None) -> List[Dict[str, Any]]`
Hybrid search combining exact match, full-text, and semantic similarity.
- Returns deduplicated results ranked by relevance
- Each result includes `match_type`, `title`, `link_url`, `download_url`, `text`, `rank`

### Question Answering

#### `answer_question_with_tiddlers(question: str, top_k: int = 5, openai_api_key: str = None, model: str = "gpt-4o-mini") -> Dict[str, Any]`
Answer questions using RAG (Retrieval-Augmented Generation) with LangChain.
- `question`: Natural language question
- `top_k`: Number of relevant tiddlers to retrieve
- `model`: OpenAI model to use (gpt-4o-mini, gpt-4o, etc.)
- Returns: Dict with `question`, `answer`, `sources` (list of tiddlers with ranks)

### Text Processing Utilities

#### `strip_html(text: str) -> str`
Remove all HTML tags and decode HTML entities.
- Removes: `<tags>`, `&entities;`
- Cleans up whitespace

#### `strip_wikitext(text: str) -> str`
Remove all TiddlyWiki wikitext markup, leaving only plain text.
- Removes: widgets, macros, transclusions, links, formatting, headers, lists, tables
- Preserves: link text and actual content

#### `split_by_headings(text: str) -> List[Dict[str, str]]`
Split wikitext by headings into sections.
- Returns: List of dicts, each with one heading as key and its content as value
- Content before first heading is stored under `_preamble`
- Heading text includes the `!` markers

### Utility Functions

#### `title_to_link_path(title: str) -> str`
Convert tiddler title to URL hash path (e.g., `#MyTiddler`).

#### `title_to_download_path(title: str) -> str`
Convert tiddler title to API download path.

#### `delete_all_embeddings(table_name: str = 'tiddlers') -> None`
Delete all tiddler records from the database.

## Project Structure

The `serve` command automatically creates and populates three TiddlyWiki folders:

```
tw-ai/
├── TW.org/              # Official TiddlyWiki docs (auto-created fresh each time)
├── GrokTW/              # Grok TiddlyWiki tutorials (auto-created fresh each time)
├── DevTW/               # TiddlyWiki dev docs (auto-created fresh each time)
├── TW Hidden Settings/  # Custom tiddlers to inject into each wiki (optional)
├── tiddlywiki_api.py
├── .env
└── ...
```

**Automatic Setup:**

When you run `python tiddlywiki_api.py serve`, the command will:
1. **Delete** existing TW.org, GrokTW, and DevTW folders (ensuring clean slate)
2. **Download** the latest HTML from each website:
   - TW.org from https://tiddlywiki.com
   - GrokTW from https://groktiddlywiki.com
   - DevTW from https://tiddlywiki.com/dev
3. **Import** each HTML file into its respective folder using TiddlyWiki's `--savewikifolder` command
4. **Copy** all files from "TW Hidden Settings" folder into each wiki's `/tiddlers` subfolder (if the folder exists)
5. **Start** local servers for each instance

**TW Hidden Settings Folder (Optional):**

If you want to inject custom tiddlers into all three wikis (e.g., custom CSS, configuration, or hidden tiddlers), create a folder named "TW Hidden Settings" in the project root and add your tiddler files there. These will be automatically copied into each wiki's tiddlers folder during the serve process.

This means you don't need to manually set up these folders - just run the `serve` command and it will handle everything!

## Requirements

- **Python 3.8+**
- **PostgreSQL 12+** with pgvector extension
- **OpenAI API key** for embeddings and question answering
- **TiddlyWiki** (Node.js version) for running local servers

### Python Dependencies

All dependencies are listed in `requirements.txt`:

```
requests
python-dotenv
psycopg2-binary
pgvector
langchain-openai
```

The `langchain-openai` package automatically includes required dependencies like `langchain` and `langchain-core`.

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Common Use Cases

### 1. Complete Workflow
```python
from tiddlywiki_api import get_tiddlers_with_embeddings, save_tiddlers_to_postgres
from dotenv import load_dotenv

load_dotenv()

# First, start servers with: python tiddlywiki_api.py serve
# Then index all local TiddlyWiki instances

wikis = [
    ("127.0.0.1:8081", "https://tiddlywiki.com", None),
    ("127.0.0.1:8082", "https://groktiddlywiki.com", "[all[shadows]!prefix[$]]"),
    ("127.0.0.1:8083", "https://tiddlywiki.com/dev", None)
]

all_tiddlers = []
for scan_domain, link_domain, filter_expr in wikis:
    print(f"Fetching from {scan_domain}...")
    tiddlers = get_tiddlers_with_embeddings(scan_domain, link_domain, filter_expr)
    all_tiddlers.extend(tiddlers)

save_tiddlers_to_postgres(all_tiddlers)
print(f"Indexed {len(all_tiddlers)} tiddlers total")
```

### 2. Search with Filters
```python
# Only index non-system tiddlers
tiddlers = get_tiddlers_with_embeddings(
    scan_domain="localhost:8080",
    link_domain="example.com",
    filter="[!is[system]!is[shadow]]"
)
```

### 3. Build a Custom Search Interface
```python
from tiddlywiki_api import search_similar_tiddlers_with_text

def search_handler(query: str):
    results = search_similar_tiddlers_with_text(query, top_k=10)

    for result in results:
        print(f"[{result['match_type']}] {result['title']}")
        print(f"Score: {result['rank']:.3f}")
        print(f"Preview: {result['text'][:200]}...")
        print(f"Link: {result['link_url']}\n")
```

### 4. Process Tiddlers by Section
```python
from tiddlywiki_api import get_tiddler_content, split_by_headings, strip_wikitext

# Get a tiddler
tiddler = get_tiddler_content("localhost:8080", "MyTiddler")

# Split into sections
sections = split_by_headings(tiddler['text'])

# Process each section
for section in sections:
    for heading, content in section.items():
        clean_content = strip_wikitext(content)
        print(f"{heading}: {len(clean_content)} characters")
```

## Troubleshooting

### TiddlyWiki Server Issues
If the `serve` command fails:
- **TiddlyWiki not found**: Install TiddlyWiki globally: `npm install -g tiddlywiki`
- **Download fails**: Check your internet connection or if the websites are accessible
- **Port already in use**: Check if ports 8081-8083 are available with `netstat -an | grep 808`
- **Permission denied**: Ensure you have read/write access to create folders in the project directory
- **Import errors**: The command will continue with existing content if import fails

### Database Connection Issues
If you get connection errors:
1. Ensure PostgreSQL is running
2. Verify credentials in `.env` file
3. Check that pgvector extension is installed:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

### OpenAI API Errors
- **Rate limits**: Reduce batch size or add delays between requests
- **Invalid API key**: Check `OPENAI_API_KEY` in `.env`
- **Model not found**: Ensure you're using a valid model name (gpt-4o-mini, gpt-4o, etc.)

### Empty Search Results
- Check that tiddlers were indexed: `SELECT COUNT(*) FROM tiddlers;`
- Verify embeddings exist: `SELECT COUNT(*) FROM tiddlers WHERE embedding IS NOT NULL;`
- Try different search methods (exact, full-text, semantic) to diagnose

### Performance Issues
- **Slow similarity search**: Ensure the vector index is created (automatic on first save)
- **Large datasets**: Consider increasing `lists` parameter in IVFFlat index
- **Memory usage**: Process tiddlers in batches rather than all at once

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
