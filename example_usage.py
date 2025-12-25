"""
Example script demonstrating TiddlyWiki semantic search workflow.

This script shows how to:
1. Fetch tiddlers from a TiddlyWiki instance
2. Generate embeddings for their content
3. Save to PostgreSQL with pgvector
4. Perform semantic searches
"""

from tiddlywiki_api import (
    get_tiddlers_with_embeddings,
    save_tiddlers_to_postgres,
    search_similar_tiddlers
)
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()

    # Configuration
    TIDDLYWIKI_DOMAIN = "127.0.0.1:8080"  # Change to your TiddlyWiki instance

    # Check if database is configured
    if not os.getenv('POSTGRES_DB'):
        print("Error: PostgreSQL environment variables not configured.")
        print("Please copy .env.example to .env and fill in your database credentials.")
        return

    # Step 1: Fetch tiddlers with embeddings
    print(f"Step 1: Fetching tiddlers from {TIDDLYWIKI_DOMAIN}...")
    try:
        tiddlers = get_tiddlers_with_embeddings(TIDDLYWIKI_DOMAIN)
        print(f"✓ Successfully fetched {len(tiddlers)} tiddlers\n")
    except Exception as e:
        print(f"✗ Error fetching tiddlers: {e}")
        return

    # Step 2: Save to PostgreSQL
    print("Step 2: Saving tiddlers to PostgreSQL...")
    try:
        save_tiddlers_to_postgres(tiddlers)
        print("✓ Successfully saved to database\n")
    except Exception as e:
        print(f"✗ Error saving to database: {e}")
        return

    # Step 3: Perform semantic searches
    print("Step 3: Performing semantic searches...")

    search_queries = [
        "artificial intelligence and machine learning",
        "project management techniques",
        "python programming tips"
    ]

    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        print("-" * 60)

        try:
            results = search_similar_tiddlers(query, top_k=3)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Similarity: {result['similarity']:.4f}")
                    print(f"   URL: {result['link_url']}")
            else:
                print("No results found.")

        except Exception as e:
            print(f"✗ Error searching: {e}")

    print("\n" + "=" * 60)
    print("Example completed! You can now use search_similar_tiddlers()")
    print("in your own scripts to find relevant tiddlers.")

if __name__ == '__main__':
    main()
