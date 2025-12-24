import requests
from typing import List, Dict, Any
from urllib.parse import quote
from langchain_openai import OpenAIEmbeddings


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


def get_tiddlers_with_embeddings(domain: str, openai_api_key: str = None) -> List[Dict[str, Any]]:
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
    tiddlers = get_tiddlers(domain)

    # Fetch full content for each tiddler
    tiddler_data = []
    texts = []

    for tiddler in tiddlers:
        title = tiddler['title']
        try:
            full_tiddler = get_tiddler_content(domain, title)
            text_content = full_tiddler.get('text', '')

            # Build the full URL
            if not domain.startswith(('http://', 'https://')):
                domain = f'http://{domain}'
            full_url = f"{domain.rstrip('/')}{title_to_link_path(title)}"

            tiddler_data.append({
                'title': title,
                'url': full_url,
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
            'url': tiddler['url'],
            'embedding': embeddings[i]
        })

    return results


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 1:
        print("Usage: python tiddlywiki_api.py <domain>")
        print("Example: python tiddlywiki_api.py localhost:8080")
        sys.exit(1)

    if len(sys.argv) < 1:
        domain = sys.argv[1]
    else:
        domain = "127.0.0.1:8080"
    try:
        tiddlers = get_tiddlers(domain)
        print(f"Found {len(tiddlers)} tiddlers:")
        for tiddler in tiddlers:
            print(f"  - {tiddler.get('title', 'Untitled')} (modified: {tiddler.get('modified', 'N/A')})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
