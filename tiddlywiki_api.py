import requests
from typing import List, Dict, Any
from urllib.parse import quote
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import os
import re
import html


def strip_html(text: str) -> str:
    """
    Remove all HTML tags and entities from a string.

    Args:
        text: The string containing HTML markup

    Returns:
        A clean string with all HTML tags removed and HTML entities decoded

    Example:
        >>> strip_html("<p>Hello &amp; welcome!</p>")
        'Hello & welcome!'
        >>> strip_html("<div>Line 1<br>Line 2</div>")
        'Line 1 Line 2'
        >>> strip_html("Plain text")
        'Plain text'
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities (e.g., &amp; -> &, &lt; -> <, &nbsp; -> space)
    text = html.unescape(text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    return text.strip()


def strip_wikitext(text: str) -> str:
    """
    Remove all TiddlyWiki wikitext markup from a string, leaving only plain text.

    Args:
        text: The string containing TiddlyWiki wikitext markup

    Returns:
        A clean string with all wikitext syntax removed

    Example:
        >>> strip_wikitext("''bold'' and //italic// text")
        'bold and italic text'
        >>> strip_wikitext("[[Link to Tiddler]]")
        'Link to Tiddler'
        >>> strip_wikitext("[[Displayed Text|Target]]")
        'Displayed Text'
        >>> strip_wikitext("! Header\\n\\nNormal text")
        'Header Normal text'
    """
    if not text:
        return ""

    # Remove HTML/XML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove TiddlyWiki widgets (e.g., <$widget>...</$widget>)
    text = re.sub(r'<\$[^>]*>.*?</\$[^>]*>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\$[^>]*/?>', '', text)

    # Remove macros (e.g., <<macro param>>)
    text = re.sub(r'<<[^>]+>>', '', text)

    # Remove transclusions (e.g., {{tiddler}}, {{tiddler||template}})
    text = re.sub(r'\{\{[^}]+\}\}', '', text)

    # Remove images (e.g., [img[url]], [img[tooltip|url]])
    text = re.sub(r'\[img\[[^\]]+\]\]', '', text)

    # Extract text from external links (e.g., [ext[text|url]] -> text)
    text = re.sub(r'\[ext\[([^\]|]+)\|[^\]]+\]\]', r'\1', text)
    text = re.sub(r'\[ext\[([^\]]+)\]\]', r'\1', text)

    # Extract text from internal links with custom display text (e.g., [[Display|Target]] -> Display)
    text = re.sub(r'\[\[([^\]|]+)\|[^\]]+\]\]', r'\1', text)

    # Extract text from simple internal links (e.g., [[Target]] -> Target)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Remove code blocks (multi-line)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)

    # Remove formatting markers
    text = re.sub(r"'''", '', text)  # Bold (triple quotes)
    text = re.sub(r"''", '', text)   # Bold (double quotes)
    text = re.sub(r'//', '', text)   # Italic
    text = re.sub(r'__', '', text)   # Underline
    text = re.sub(r'~~', '', text)   # Strikethrough
    text = re.sub(r'\^\^', '', text) # Superscript
    text = re.sub(r',,', '', text)   # Subscript

    # Remove headers (e.g., ! !! !!!)
    text = re.sub(r'^!+\s*', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)

    # Remove list markers (*, #, ;, :)
    text = re.sub(r'^\s*[\*#;:]+\s*', '', text, flags=re.MULTILINE)

    # Remove blockquote markers (<<<)
    text = re.sub(r'^<<<+\s*', '', text, flags=re.MULTILINE)

    # Remove table syntax (|cell|cell|)
    text = re.sub(r'\|', ' ', text)

    # Remove definition lists markers
    text = re.sub(r'^;', '', text, flags=re.MULTILINE)
    text = re.sub(r'^:', '', text, flags=re.MULTILINE)

    # Remove remaining brackets and braces
    text = re.sub(r'[\[\]\{\}]', '', text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    return text.strip()


def split_by_headings(text: str) -> List[Dict[str, str]]:
    """
    Split TiddlyWiki wikitext by headings into sections.

    Args:
        text: The string containing TiddlyWiki wikitext with headings

    Returns:
        A list of dictionaries, where each dict has one key-value pair:
        - Key: The original heading line (e.g., "! Main Heading")
        - Value: All wikitext content between that heading (including the heading itself) and the next (or end of string)

    Example:
        >>> text = "! Heading 1\\nContent 1\\n!! Heading 2\\nContent 2"
        >>> result = split_by_headings(text)
        >>> result
        [{'! Heading 1': '! Heading 1\nContent 1\\n'}, {'!! Heading 2': '!! Heading 2\nContent 2'}]

        >>> text = "Intro text\\n! Section 1\\nText here\\n! Section 2\\nMore text"
        >>> result = split_by_headings(text)
        >>> result
        [{'_preamble': 'Intro text\\n'}, {'! Section 1': '! Section 1\nText here\\n'}, {'! Section 2': '! Section 2\nMore text'}]
    """
    if not text:
        return []

    # Split into lines
    lines = text.split('\n')

    sections = []
    current_heading = None
    current_content = []

    for line in lines:
        # Check if line is a heading (starts with !)
        if re.match(r'^!+\s+', line):
            # Save previous section if it exists
            if current_heading is not None:
                content = '\n'.join(current_content)
                sections.append({current_heading: content})
            elif current_content:
                # Content before first heading (preamble)
                content = '\n'.join(current_content)
                sections.append({'_preamble': content})

            # Start new section
            current_heading = line
            current_content = [line]
        else:
            # Add line to current content
            current_content.append(line)

    # Don't forget the last section
    if current_heading is not None:
        content = '\n'.join(current_content)
        sections.append({current_heading: content})
    elif current_content:
        # If no headings at all, return content under '_preamble'
        content = '\n'.join(current_content)
        sections.append({'_preamble': content})

    return sections


def get_tiddlers(domain: str, filter: str = None) -> List[Dict[str, Any]]:
    """
    Fetch a list of tiddlers from a TiddlyWiki instance.

    Args:
        domain: The domain of the TiddlyWiki instance (e.g., "example.com" or "http://example.com")
        filter: An optional filter that should be sent to the scan_domain to filter tiddlers

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
    if filter is not None:
        url += f"?filter={filter}"

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


def get_tiddlers_with_embeddings(scan_domain: str, link_domain: str, filter: str = None, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all tiddlers and generate OpenAI embeddings for their text content.

    Args:
        scan_domain: The domain of the TiddlyWiki instance to be scanned
        link_domain: The domain that the user should be linked to when displaying results
        filter: An optional filter that should be sent to the scan_domain to filter tiddlers
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
    tiddlers = get_tiddlers(scan_domain, filter)

    # Fetch full content for each tiddler
    tiddler_data = []
    texts = []

    for tiddler in tiddlers:
        title = tiddler['title']
        try:
            full_tiddler = get_tiddler_content(scan_domain, title)
            text_content = strip_html(strip_wikitext(full_tiddler.get('text', '')))

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
            'text': tiddler['text']
        })

    return results


def get_tiddler_section_embeddings(scan_domain: str, link_domain: str, filter: str = None, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all tiddlers and generate OpenAI embeddings for their text content.

    Args:
        scan_domain: The domain of the TiddlyWiki instance to be scanned
        link_domain: The domain that the user should be linked to when displaying results
        filter: An optional filter that should be sent to the scan_domain to filter tiddlers
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
        >>> results = get_tiddler_section_embeddings("localhost:8080")
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
    tiddler_results = []
    for i, tiddler in enumerate(tiddler_data):
        tiddler_result = {
            'title': tiddler['title'],
            'link_url': tiddler['link_url'],
            'download_url': tiddler['download_url'],
            'embedding': embeddings[i],
            'text': strip_html(tiddler['text'])
        }
        tiddler_results.append(tiddler_result)

    return results


def exact_search(query: str, top_k: int = 5):
    """Exact keyword matching for specific terms, names, dates"""
        
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
        search_query = """
        SELECT
            'EXACT' AS match_type,
            title,
            link_url,
            download_url,
            text,
            1.0 AS rank
        FROM tiddlers
        WHERE text ILIKE %s
        ORDER BY char_length(text)  -- Prefer shorter, more focused articles
        LIMIT %s
        """

        cur.execute(search_query, (f"%{query}%", top_k))
        rows = cur.fetchall()

        # Format results
        results = []
        for row in rows:
            results.append({
                'match_type': row[0],
                'title': row[1],
                'link_url': row[2],
                'download_url': row[3],
                'text': row[4],
                'rank': float(row[5])
            })

        return results

    finally:
        cur.close()
        conn.close()


def full_text_search(query: str, top_k: int = 5):
    """Full text matching for specific terms, names, dates"""
        
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
        search_query = """
        SELECT
            'FULLTEXT' AS match_type,
            title,
            link_url,
            download_url,
            text,
            ts_rank_cd(
                to_tsvector('english', text),
                plainto_tsquery('english', %s)
            )  AS rank
        FROM tiddlers,
            plainto_tsquery('english', %s) query
        WHERE to_tsvector('english', text) @@ query
        ORDER BY rank DESC
        LIMIT %s
        """

        cur.execute(search_query, (query, query, top_k))
        rows = cur.fetchall()

        # Format results
        results = []
        for row in rows:
            results.append({
                'match_type': row[0],
                'title': row[1],
                'link_url': row[2],
                'download_url': row[3],
                'text': row[4],
                'rank': float(row[5])
            })

        return results

    finally:
        cur.close()
        conn.close()


def similarity_search(query: str, top_k: int = 5, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Search for tiddlers most similar to the given query string, including their text content.

    Args:
        query: The search query string to find similar tiddlers for
        top_k: Number of top results to return (default: 5)
        openai_api_key: Optional OpenAI API key (if not set in environment)

    Returns:
        A list of dictionaries containing:
        - title: The tiddler title
        - link_url: URL to view the tiddler
        - download_url: URL to download the tiddler
        - text: The full text content of the tiddler
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
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        search_query = f"""
        SELECT
            'SIMILARITY' AS match_type,
            title,
            link_url,
            download_url,
            text,
            embedding <-> %s AS rank
        FROM tiddlers
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> %s ASC
        LIMIT %s;
        """

        cur.execute(search_query, (embedding_str, embedding_str, top_k))
        rows = cur.fetchall()

        # Format results
        results = []
        for row in rows:
            results.append({
                'match_type': row[0],
                'title': row[1],
                'link_url': row[2],
                'download_url': row[3],
                'text': row[4],
                'rank': float(row[5])
            })

        return results

    finally:
        cur.close()
        conn.close()


def combine_search_results(sets_of_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen_urls = set()
    results = []
    for result_set in sets_of_results:
        for result in result_set:
            link_url = result["link_url"]
            if link_url in seen_urls:
                continue
            results.append(result)
            seen_urls.add(link_url)
    print(f"Returning {len(results)} results from combine_search_results")
    return sorted(results, key=lambda r: r["rank"], reverse=True)


def search_similar_tiddlers_with_text(query: str, top_k: int = 5, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Search for tiddlers most similar to the given query string, including their text content.

    Args:
        query: The search query string to find similar tiddlers for
        top_k: Number of top results to return (default: 5)
        openai_api_key: Optional OpenAI API key (if not set in environment)

    Returns:
        A list of dictionaries containing:
        - title: The tiddler title
        - link_url: URL to view the tiddler
        - download_url: URL to download the tiddler
        - text: The full text content of the tiddler
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
    """
    exact_match_results = exact_search(query, top_k)
    print(f"Found {len(exact_match_results)} with an exact match")
    full_text_results = full_text_search(query, top_k)
    print(f"Found {len(full_text_results)} with a full text match")
    similarity_results = similarity_search(query, top_k, openai_api_key)
    print(f"Found {len(similarity_results)} with similarity match")

    final_results = combine_search_results([exact_match_results, full_text_results, similarity_results])

    return final_results


def answer_question_with_tiddlers(question: str, top_k: int = 5, openai_api_key: str = None, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Answer a question using relevant tiddlers from the database with LangChain v1.

    This function performs semantic search to find the most relevant tiddlers,
    then uses LangChain to generate an answer based on their content.

    Args:
        question: The question to answer
        top_k: Number of top relevant tiddlers to use (default: 5)
        openai_api_key: Optional OpenAI API key (if not set in environment)
        model: OpenAI model to use (default: "gpt-4o-mini")

    Returns:
        A dictionary containing:
        - question: The original question
        - answer: The generated answer
        - sources: List of tiddlers used as sources

    Environment Variables Required:
        POSTGRES_HOST: Database host (default: localhost)
        POSTGRES_PORT: Database port (default: 5432)
        POSTGRES_DB: Database name
        POSTGRES_USER: Database user
        POSTGRES_PASSWORD: Database password
        OPENAI_API_KEY: OpenAI API key (if not passed as parameter)

    Raises:
        ValueError: If required environment variables are missing
        Exception: If search or LLM operations fail

    Example:
        >>> load_dotenv()
        >>> result = answer_question_with_tiddlers("How do I create a new tiddler?")
        >>> print(result['answer'])
        >>> for source in result['sources']:
        ...     print(f"- {source['title']}")
    """
    # Search for relevant tiddlers
    print(f"Searching for relevant tiddlers...")
    tiddlers = search_similar_tiddlers_with_text(question, top_k=top_k, openai_api_key=openai_api_key)

    if not tiddlers:
        return {
            'question': question,
            'answer': "I couldn't find any relevant tiddlers to answer your question.",
            'sources': []
        }

    # Prepare context from tiddlers
    context_parts = []
    for i, tiddler in enumerate(tiddlers, 1):
        context_parts.append(
            f"Source {i}: {tiddler['title']}\n"
            f"URL: {tiddler['link_url']}\n"
            f"Content: {tiddler['text']}\n"
        )
    context = "\n---\n".join(context_parts)

    # Create LangChain prompt template
    template = """You are a helpful assistant that answers questions based on the provided TiddlyWiki tiddlers.

Use the following tiddler contents to answer the question. If the tiddlers don't contain enough information to answer the question, say so clearly.

Question: {question}

Tiddler Contents:
{context}

Please provide a clear, comprehensive answer based on the information above. If you reference specific information, mention which tiddler it came from with both its title and link_url."""

    prompt = ChatPromptTemplate.from_template(template)

    # Initialize LangChain components
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=openai_api_key
    )
    output_parser = StrOutputParser()

    # Create the chain using LCEL (LangChain Expression Language)
    chain = (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | output_parser
    )

    # Generate answer
    print(f"Generating answer using {model}...")
    answer = chain.invoke({"context": context, "question": question})

    # Prepare sources
    sources = [
        {
            'title': t['title'],
            'link_url': t['link_url'],
            'rank': t['rank']
        }
        for t in tiddlers
    ]

    return {
        'question': question,
        'answer': answer,
        'sources': sources
    }


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

            results = search_similar_tiddlers_with_text(query, top_k=top_k, openai_api_key=OPENAI_API_KEY)

            if results:
                print(f"Found {len(results)} similar tiddlers:\n")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Similarity: {result['rank']:.4f}")
                    print(f"   URL: https://{result['link_url']}")
                    print(f"   Text: {result['text'][:100]}")
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

    elif len(sys.argv) > 1 and sys.argv[1] == 'ask':
        # Ask mode - answer questions using tiddlers
        if len(sys.argv) < 3:
            print("Usage: python tiddlywiki_api.py ask <question> [top_k] [model]")
            print("Example: python tiddlywiki_api.py ask 'How do I create a tiddler?' 5 gpt-4o-mini")
            print("\nAvailable models: gpt-4o-mini (default), gpt-4o, gpt-4-turbo, gpt-3.5-turbo")
            sys.exit(1)

        question = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        model = sys.argv[4] if len(sys.argv) > 4 else "gpt-4o-mini"

        try:
            print(f"Question: {question}")
            print(f"Using top {top_k} relevant tiddlers")
            print(f"Model: {model}\n")

            result = answer_question_with_tiddlers(
                question=question,
                top_k=top_k,
                openai_api_key=OPENAI_API_KEY,
                model=model
            )

            print("\n" + "="*80)
            print("ANSWER:")
            print("="*80)
            print(result['answer'])
            print("\n" + "="*80)
            print("SOURCES:")
            print("="*80)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']}")
                print(f"   Rank: {source['rank']:.4f}")
                print(f"   URL: {source['link_url']}")
                print()

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
                print("  python tiddlywiki_api.py ask 'your question' [top_k] [model]")
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

            all_tiddlers = []
            for scan_domain, link_domain, filter in [("127.0.0.1:8081", "www.tiddlywiki.com", None), ("127.0.0.1:8082", "groktiddlywiki.com", "[all[shadows]!prefix[$]!prefix[Ex:]!prefix[Sn]!prefix[Ta]sort[title]]"), ("127.0.0.1:8083", "tiddlywiki.com/dev", None)]:
                print(f"\nScanning from {scan_domain} (links to {link_domain})...")
                tiddlers_for_domain = get_tiddlers_with_embeddings(scan_domain, link_domain, filter, openai_api_key=OPENAI_API_KEY)
                print(f"Found {len(tiddlers_for_domain)} tiddlers with embeddings from {scan_domain} (link to {link_domain})")
                all_tiddlers.extend(tiddlers_for_domain)

            # Save to PostgreSQL if environment variables are set
            print("Saving to PostgreSQL database...")

            save_tiddlers_to_postgres(all_tiddlers)
            print("\nAvailable commands:")
            print("  python tiddlywiki_api.py search 'your query' [top_k]")
            print("  python tiddlywiki_api.py delete")

            print(f"\nSample tiddlers:")
            for tiddler in all_tiddlers[:5]:  # Show first 5
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
                print("  python tiddlywiki_api.py ask 'your question' [top_k] [model]")
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
