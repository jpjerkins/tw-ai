import requests
from typing import List, Dict, Any


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


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tiddlywiki_api.py <domain>")
        print("Example: python tiddlywiki_api.py localhost:8080")
        sys.exit(1)

    domain = sys.argv[1]
    try:
        tiddlers = get_tiddlers(domain)
        print(f"Found {len(tiddlers)} tiddlers:")
        for tiddler in tiddlers:
            print(f"  - {tiddler.get('title', 'Untitled')} (modified: {tiddler.get('modified', 'N/A')})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
