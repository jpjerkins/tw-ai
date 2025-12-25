import requests
from typing import List, Dict, Any
from urllib.parse import quote
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import os


def get_tiddlers(domain: str) -> List[Dict[str, Any]]:
    """
    Fetch a list of tiddlers from a TiddlyWiki instance.

    Args:
        domain: The domain of the TiddlyWiki instance (e.g., "example.com" or "http://example.com")

    Returns:
        A list of tiddler objects, each containing metadata like title, created, modified, tags, etc.

    Raises:
        requests.RequestException: If the HTTP request fails
        ValueError: If the response is not valid JSON

    Example:
        >>> tiddlers = get_tiddlers("localhost:8080")
        >>> for tiddler in tiddlers:
        ...     print(tiddler['title'])
    """
    # Ensure domain has a protocol
    if not domain.startswith(('http://', 'https://')):
        domain = f'http://{domain}'

    # Remove trailing slash if present
    domain = domain.rstrip('/')

    # Construct the full URL
    url = f'{domain}/recipes/default/tiddlers.json'

    # Make the request
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Parse and return the JSON response
    return response.json()


def title_to_download_path(title: str) -> str:
    """
    Convert a tiddler title to a URL hash path.

    Args:
        title: The tiddler title to convert

    Returns:
        A URL-encoded hash path (e.g., "/recipes/default/tiddlers/My%20First%20Tiddler")

    Example:
        >>> title_to_download_path("My First Tiddler")
        '/recipes/default/tiddlers/My%20First%20Tiddler'
        >>> title_to_download_path("Hello World!")
        '/recipes/default/tiddlers/Hello%20World%21'
    """
    return f'/recipes/default/tiddlers/{quote(title, safe="")}'


def title_to_link_path(title: str) -> str:
    """
    Convert a tiddler title to a URL hash path.

    Args:
        title: The tiddler title to convert

    Returns:
        A URL-encoded hash path (e.g., "#My%20First%20Tiddler")

    Example:
        >>> title_to_download_path("My First Tiddler")
        '#My%20First%20Tiddler'
        >>> title_to_download_path("Hello World!")
        '#Hello%20World%21'
    """
    return f'#{quote(title, safe="")}'


def titles_to_paths(tiddlers: List[Dict[str, Any]]) -> List[str]:
    """
    Convert a list of tiddlers to a list of URL hash paths.

    Args:
        tiddlers: List of tiddler objects returned from get_tiddlers()

    Returns:
        A list of URL-encoded hash paths

    Example:
        >>> tiddlers = get_tiddlers("localhost:8080")
        >>> paths = titles_to_paths(tiddlers)
        >>> print(paths)
        ['/recipes/default/tiddlers/My%20First%20Tiddler', '/recipes/default/tiddlers/Another%20Page', ...]
    """
    return [title_to_download_path(tiddler['title']) for tiddler in tiddlers]


def get_tiddler_content(domain: str, title: str) -> Dict[str, Any]:
    """
    Fetch the full content of a single tiddler.

    Args:
        domain: The domain of the TiddlyWiki instance
        title: The title of the tiddler to fetch

    Returns:
        The full tiddler object including the text content

    Raises:
        requests.RequestException: If the HTTP request fails
    """
    # Ensure domain has a protocol
    if not domain.startswith(('http://', 'https://')):
        domain = f'http://{domain}'

    domain = domain.rstrip('/')

    # Construct the URL for the individual tiddler
    url = f'{domain}/recipes/default/tiddlers/{quote(title, safe="")}'

    response = requests.get(url)
    response.raise_for_status()

    return response.json()


def get_tiddlers_with_embeddings(scan_domain: str, link_domain: str, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all tiddlers and generate OpenAI embeddings for their text content.

    Args:
        domain: The domain of the TiddlyWiki instance
        openai_api_key: Optional OpenAI API key (if not set in environment)

    Returns:
        A list of dictionaries containing:
        - title: The tiddler title
        - url: The full URL path to the tiddler
        - embedding: The OpenAI embedding vector for the tiddler's text

    Raises:
        requests.RequestException: If any HTTP request fails
        ValueError: If OpenAI API key is not provided or set in environment

    Example:
        >>> results = get_tiddlers_with_embeddings("localhost:8080")
        >>> for item in results:
        ...     print(f"{item['title']}: {len(item['embedding'])} dimensions")
    """
    # Initialize OpenAI embeddings model
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key) if openai_api_key else OpenAIEmbeddings()

    # Get list of tiddlers
    tiddlers = get_tiddlers(scan_domain)

    # Fetch full content for each tiddler
    tiddler_data = []
    texts = []

    for tiddler in tiddlers:
        title = tiddler['title']
        try:
            full_tiddler = get_tiddler_content(scan_domain, title)
            text_content = full_tiddler.get('text', '')

            # Build the full URL
            if not scan_domain.startswith(('http://', 'https://')):
                scan_domain = f'http://{scan_domain}/'
            link_url = f"{link_domain.rstrip('/')}/{title_to_link_path(title).lstrip('/')}"
            download_url = f"{scan_domain.rstrip('/')}/{title_to_download_path(title).lstrip('/')}"

            tiddler_data.append({
                'title': title,
                'link_url': link_url,
                'download_url': download_url,
                'text': text_content
            })
            texts.append(text_content)
        except Exception as e:
            print(f"Warning: Could not fetch tiddler '{title}': {e}")
            continue

    # Generate embeddings for all texts at once
    embeddings = embeddings_model.embed_documents(texts)

    # Combine the data with embeddings
    results = []
    for i, tiddler in enumerate(tiddler_data):
        results.append({
            'title': tiddler['title'],
            'link_url': tiddler['link_url'],
            'download_url': tiddler['download_url'],
            'embedding': embeddings[i],
            'text': text_content
        })

    return results


def search_similar_tiddlers(query: str, top_k: int = 5, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Search for tiddlers most similar to the given query string using semantic search.

    Args:
        query: The search query string to find similar tiddlers for
        top_k: Number of top results to return (default: 5)
        openai_api_key: Optional OpenAI API key (if not set in environment)

    Returns:
        A list of dictionaries containing:
        - title: The tiddler title
        - link_url: URL to view the tiddler
        - download_url: URL to download the tiddler
        - similarity: Cosine similarity score (0-1, higher is more similar)

    Environment Variables Required:
        POSTGRES_HOST: Database host (default: localhost)
        POSTGRES_PORT: Database port (default: 5432)
        POSTGRES_DB: Database name
        POSTGRES_USER: Database user
        POSTGRES_PASSWORD: Database password
        OPENAI_API_KEY: OpenAI API key (if not passed as parameter)

    Raises:
        psycopg2.Error: If database connection or query fails
        ValueError: If required environment variables are missing

    Example:
        >>> load_dotenv()
        >>> results = search_similar_tiddlers("machine learning concepts", top_k=3)
        >>> for result in results:
        ...     print(f"{result['title']}: {result['similarity']:.3f}")
    """
    # Initialize OpenAI embeddings model
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key) if openai_api_key else OpenAIEmbeddings()

    # Generate embedding for the query
    query_embedding = embeddings_model.embed_query(query)

    # Get connection parameters from environment variables
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }

    # Check for required environment variables
    if not all([db_config['database'], db_config['user'], db_config['password']]):
        raise ValueError(
            "Missing required environment variables: POSTGRES_DB, POSTGRES_USER, and/or POSTGRES_PASSWORD"
        )

    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    try:
        # Query for similar tiddlers using cosine similarity
        # The <=> operator calculates cosine distance, so we use 1 - distance for similarity
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        search_query = f"""
        SELECT
            title,
            link_url,
            download_url,
            embedding <-> %s AS similarity,
            text
        FROM tiddlers
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> %s
        LIMIT %s;
        """

        cur.execute(search_query, (embedding_str, embedding_str, top_k))
        rows = cur.fetchall()

        # Format results
        results = []
        for row in rows:
            results.append({
                'title': row[0],
                'link_url': row[1],
                'download_url': row[2],
                'similarity': float(row[3])
            })

        return results

    finally:
        cur.close()
        conn.close()


def delete_all_embeddings(table_name: str = 'tiddlers') -> None:
    """
    Delete all tiddler embeddings from the PostgreSQL database.

    Args:
        table_name: Name of the table to clear (default: 'tiddlers')

    Environment Variables Required:
        POSTGRES_HOST: Database host (default: localhost)
        POSTGRES_PORT: Database port (default: 5432)
        POSTGRES_DB: Database name
        POSTGRES_USER: Database user
        POSTGRES_PASSWORD: Database password

    Raises:
        psycopg2.Error: If database connection or operations fail
        ValueError: If required environment variables are missing

    Example:
        >>> load_dotenv()
        >>> delete_all_embeddings()
        Successfully deleted all embeddings from table 'tiddlers'
    """
    # Get connection parameters from environment variables
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }

    # Check for required environment variables
    if not all([db_config['database'], db_config['user'], db_config['password']]):
        raise ValueError(
            "Missing required environment variables: POSTGRES_DB, POSTGRES_USER, and/or POSTGRES_PASSWORD"
        )

    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    try:
        # Get count before deletion
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cur.fetchone()[0]

        # Delete all records
        cur.execute(f"DELETE FROM {table_name};")

        # Commit the transaction
        conn.commit()
        print(f"Successfully deleted {count} tiddler(s) from table '{table_name}'")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def save_tiddlers_to_postgres(tiddlers: List[Dict[str, Any]], table_name: str = 'tiddlers') -> None:
    """
    Save tiddlers with embeddings to a PostgreSQL database using pgvector.

    Args:
        tiddlers: List of tiddler dictionaries from get_tiddlers_with_embeddings()
                 Each should contain: title, link_url, download_url, embedding
        table_name: Name of the table to store the data (default: 'tiddlers')

    Environment Variables Required:
        POSTGRES_HOST: Database host (default: localhost)
        POSTGRES_PORT: Database port (default: 5432)
        POSTGRES_DB: Database name
        POSTGRES_USER: Database user
        POSTGRES_PASSWORD: Database password

    Raises:
        psycopg2.Error: If database connection or operations fail
        KeyError: If required environment variables are missing

    Example:
        >>> load_dotenv()
        >>> tiddlers = get_tiddlers_with_embeddings("localhost:8080")
        >>> save_tiddlers_to_postgres(tiddlers)
        Saved 42 tiddlers to PostgreSQL database
    """
    # Get connection parameters from environment variables
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }

    # Check for required environment variables
    if not all([db_config['database'], db_config['user'], db_config['password']]):
        raise ValueError(
            "Missing required environment variables: POSTGRES_DB, POSTGRES_USER, and/or POSTGRES_PASSWORD"
        )

    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    try:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Determine embedding dimension from first tiddler
        embedding_dim = len(tiddlers[0]['embedding']) if tiddlers else 1536  # Default to OpenAI's dimension

        # Create table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            title TEXT UNIQUE NOT NULL,
            link_url TEXT NOT NULL,
            download_url TEXT NOT NULL,
            embedding vector({embedding_dim}),
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
        ON {table_name} USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        cur.execute(create_table_query)

        # Prepare data for insertion
        insert_query = f"""
        INSERT INTO {table_name} (title, link_url, download_url, embedding, text)
        VALUES %s
        ON CONFLICT (title)
        DO UPDATE SET
            link_url = EXCLUDED.link_url,
            download_url = EXCLUDED.download_url,
            embedding = EXCLUDED.embedding,
            text = EXCLUDED.text,
            updated_at = CURRENT_TIMESTAMP;
        """

        # Convert embeddings to the format expected by psycopg2
        values = [
            (
                tiddler['title'],
                tiddler['link_url'],
                tiddler['download_url'],
                tiddler['embedding'],
                tiddler['text']
            )
            for tiddler in tiddlers
        ]

        # Execute batch insert
        execute_values(cur, insert_query, values)

        # Commit the transaction
        conn.commit()
        print(f"Successfully saved {len(tiddlers)} tiddlers to PostgreSQL database (table: {table_name})")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    # Example usage
    import sys

    # Load environment variables from .env
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Check if user wants to search
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        # Search mode
        if len(sys.argv) < 3:
            print("Usage: python tiddlywiki_api.py search <query> [top_k]")
            print("Example: python tiddlywiki_api.py search 'machine learning' 5")
            sys.exit(1)

        query = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

        try:
            print(f"Searching for: '{query}'")
            print(f"Retrieving top {top_k} results...\n")

            results = search_similar_tiddlers(query, top_k=top_k, openai_api_key=OPENAI_API_KEY)

            if results:
                print(f"Found {len(results)} similar tiddlers:\n")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Similarity: {result['similarity']:.4f}")
                    print(f"   URL: {result['link_url']}")
                    print()
            else:
                print("No results found.")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == 'delete':
        # Delete mode
        try:
            print("WARNING: This will delete all embeddings from the database!")
            confirm = input("Are you sure you want to continue? (yes/no): ")

            if confirm.lower() in ['yes', 'y']:
                delete_all_embeddings()
                print("\nAll embeddings have been deleted successfully.")
            else:
                print("Operation cancelled.")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif len(sys.argv) > 1 and sys.argv[1] == 'scan':
        # Fetch and save mode
        if len(sys.argv) < 2:
            print("Usage: python tiddlywiki_api.py scan <scan_domain> <link_domain>")
            print("Example: python tiddlywiki_api.py scan '127.0.0.1:8081', 'www.tiddlywiki.org")
            sys.exit(1)
        scan_domain = sys.argv[2]
        link_domain = sys.argv[3]

        try:
            print(f"Fetching tiddlers from {scan_domain}...")
            tiddlers = get_tiddlers_with_embeddings(scan_domain, link_domain, openai_api_key=OPENAI_API_KEY)
            print(f"Found {len(tiddlers)} tiddlers with embeddings")

            # Save to PostgreSQL if environment variables are set
            if os.getenv('POSTGRES_DB'):
                print("Saving to PostgreSQL database...")
                save_tiddlers_to_postgres(tiddlers)
                print("\nAvailable commands:")
                print("  python tiddlywiki_api.py search 'your query' [top_k]")
                print("  python tiddlywiki_api.py delete")
            else:
                print("\nTo save to PostgreSQL, set the following environment variables:")
                print("  POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
                print("  POSTGRES_HOST (optional, default: localhost)")
                print("  POSTGRES_PORT (optional, default: 5432)")

            print(f"\nSample tiddlers:")
            for tiddler in tiddlers[:5]:  # Show first 5
                print(f"  - {tiddler['title']} ({len(tiddler['embedding'])} dimensions)")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif len(sys.argv) > 1 and sys.argv[1] == 'reindex':
        # Reindex mode
        try:
            if not os.getenv('POSTGRES_DB'):
                print("\nTo save to PostgreSQL, set the following environment variables:")
                print("  POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
                print("  POSTGRES_HOST (optional, default: localhost)")
                print("  POSTGRES_PORT (optional, default: 5432)")
                sys.exit(1)

            print("WARNING: This will delete all embeddings from the database!")
            confirm = input("Are you sure you want to continue? (yes/no): ")

            if confirm.lower() in ['yes', 'y']:
                delete_all_embeddings()
                print("All embeddings have been deleted successfully.")
            else:
                print("Operation cancelled.")

            for scan_domain, link_domain in [("127.0.0.1:8081", "www.tiddlywiki.com"), ("127.0.0.1:8082", "groktiddlywiki.com"), ("127.0.0.1:8083", "tiddlywiki.com/dev")]:
                print(f"\nScanning from {scan_domain} (links to {link_domain})...")
                tiddlers = get_tiddlers_with_embeddings(scan_domain, link_domain, openai_api_key=OPENAI_API_KEY)
                print(f"Found {len(tiddlers)} tiddlers with embeddings from {scan_domain} (link to {link_domain})")

                # Save to PostgreSQL if environment variables are set
                print("Saving to PostgreSQL database...")

            save_tiddlers_to_postgres(tiddlers)
            print("\nAvailable commands:")
            print("  python tiddlywiki_api.py search 'your query' [top_k]")
            print("  python tiddlywiki_api.py delete")

            print(f"\nSample tiddlers:")
            for tiddler in tiddlers[:5]:  # Show first 5
                print(f"  - {tiddler['title']} ({len(tiddler['embedding'])} dimensions)")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        domain = "127.0.0.1:8080"

        try:
            print(f"Fetching tiddlers from {domain}...")
            tiddlers = get_tiddlers_with_embeddings(domain, domain, openai_api_key=OPENAI_API_KEY)
            print(f"Found {len(tiddlers)} tiddlers with embeddings")

            # Save to PostgreSQL if environment variables are set
            if os.getenv('POSTGRES_DB'):
                print("Saving to PostgreSQL database...")
                save_tiddlers_to_postgres(tiddlers)
                print("\nAvailable commands:")
                print("  python tiddlywiki_api.py search 'your query' [top_k]")
                print("  python tiddlywiki_api.py delete")
            else:
                print("\nTo save to PostgreSQL, set the following environment variables:")
                print("  POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
                print("  POSTGRES_HOST (optional, default: localhost)")
                print("  POSTGRES_PORT (optional, default: 5432)")

            print(f"\nSample tiddlers:")
            for tiddler in tiddlers[:5]:  # Show first 5
                print(f"  - {tiddler['title']} ({len(tiddler['embedding'])} dimensions)")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
