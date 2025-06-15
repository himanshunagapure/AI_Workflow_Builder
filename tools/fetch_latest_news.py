import requests
import re
import os

def fetch_latest_news(query: str) -> str:
    """
    Fetch latest news based on user query and return formatted text response.
    
    This function automatically parses the user query to extract:
    1. The number of articles requested (e.g., "top 5", "top 10", "latest 3")
    2. The main search topic
    3. What type of content to return (headlines, news descriptions, etc.)
    
    If no number is specified, defaults to 5 articles.
    
    Args:
        query (str): Natural language query for news. Examples:
            - "Fetch top 3 headlines for AAPL"
            - "Get latest news about Elon Musk"
            - "Give me top 10 trending news on AI technology"
            - "Find 5 headlines about Bitcoin"
    
    Returns:
        str: Formatted text response with numbered list of results
    
    Example:
        >>> result = fetch_latest_news("Fetch top 3 headlines for AAPL")
        >>> print(result)
        1. Apple Reports Strong Q4 Earnings
        2. AAPL Stock Rises 5% After Product Launch
        3. Apple Announces New iPhone Features
        
        >>> result = fetch_latest_news("Get latest news about Elon Musk")
        >>> print(result)
        1. Tesla CEO announces new manufacturing facility...
        2. SpaceX successfully launches another rocket mission...
        3. Musk's latest tweet sparks controversy among investors...
    
    Raises:
        ValueError: If the query is empty or invalid
    """
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Parse the query to extract number of articles, content type, and main topic
    max_results, clean_query, content_type = _parse_query(query)
    
    try:
        # NewsAPI configuration
        API_KEY = os.getenv('NEWSAPI_KEY')  # Replace with actual API key
        url = "https://newsapi.org/v2/everything"
        
        params = {
            'q': clean_query,
            'sortBy': 'publishedAt',
            'pageSize': max_results,
            'language': 'en',
            'apiKey': API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if API returned an error
        if data.get('status') != 'ok':
            return f"API Error: {data.get('message', 'Unknown error')}"
        
        articles = data.get('articles', [])
        
        if not articles:
            return "No articles found for this query."
        
        # Format response based on content type
        result_lines = []
        
        for i, article in enumerate(articles[:max_results], 1):
            if content_type == 'headlines':
                # Return only headlines/titles
                title = article.get('title', 'No title available')
                # Clean up title (remove source names that appear at the end)
                title = re.sub(r'\s*-\s*[^-]*$', '', title)
                result_lines.append(f"{i}. {title}")
            else:
                # Return news descriptions/snippets
                description = article.get('description', 'No description available')
                content = article.get('content', '')
                if content and len(content) > len(description or ''):
                    # Clean content (remove truncation markers like [+X chars])
                    content = re.sub(r'\s*\[\+\d+\s+chars?\].*', '', content)
                    # if len(content) > 800:
                    #     content = content[:797] + "..."
                    result_lines.append(f"{i}. {content}")
                elif description and description != 'No description available':
                    # Truncate long descriptions and add ellipsis
                    # if len(description) > 800:
                    #     description = description[:797] + "..."
                    result_lines.append(f"{i}. {description}")
                else:
                    # Fallback to title if no description
                    title = article.get('title', 'No title available')
                    title = re.sub(r'\s*-\s*[^-]*$', '', title)
                    result_lines.append(f"{i}. {title}")
        
        return '\n'.join(result_lines)
        
    except requests.RequestException as e:
        return f"Error fetching news: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


def _parse_query(query: str) -> tuple[int, str, str]:
    """
    Parse user query to extract number of articles, clean topic, and content type.
    
    Args:
        query (str): User's natural language query
        
    Returns:
        tuple[int, str, str]: (number_of_articles, clean_topic, content_type)
    """
    
    # Convert to lowercase for easier matching
    lower_query = query.lower().strip()
    
    # Determine content type based on keywords
    content_type = 'news'  # default
    if any(word in lower_query for word in ['headline', 'title', 'header']):
        content_type = 'headlines'
    
    # Patterns to match numbers in queries
    number_patterns = [
        r'top\s+(\d+)',           # "top 5", "top 10"
        r'latest\s+(\d+)',        # "latest 5", "latest 10" 
        r'(\d+)\s+(?:news|articles|stories|headlines)',  # "5 news", "10 articles"
        r'(?:give\s+me|get|fetch|find)\s+(?:top\s+)?(\d+)',  # "give me top 5", "get 10", "fetch 3"
    ]
    
    max_results = 5  # default
    
    # Try to find a number in the query
    for pattern in number_patterns:
        match = re.search(pattern, lower_query)
        if match:
            max_results = int(match.group(1))
            break
    
    # Clean the query by removing common prefixes and number references
    clean_patterns = [
        r'^(?:fetch|find|get|give\s+me)\s+(?:top\s+\d+\s+)?(?:headlines?\s+for\s+|news\s+about\s+|latest\s+news\s+about\s+)?',
        r'^(?:top\s+\d+\s+)?(?:headlines?\s+for\s+|news\s+about\s+|latest\s+news\s+about\s+)',
        r'^(?:latest\s+\d+\s+|top\s+\d+\s+)(?:news\s+|headlines?\s+|articles?\s+)?(?:about\s+|for\s+|on\s+)?',
        r'^(?:trending\s+news\s+on\s+|news\s+about\s+)',
        r'\s+(?:news|headlines?|articles?)$',  # remove trailing words
    ]
    
    clean_query = lower_query
    for pattern in clean_patterns:
        clean_query = re.sub(pattern, '', clean_query).strip()
    
    # If query becomes empty after cleaning, extract topic differently
    if not clean_query:
        # Look for patterns like "about X", "for X", "on X"
        topic_patterns = [
            r'(?:about|for|on)\s+(.+?)(?:\s+(?:news|headlines?|articles?))?$',
            r'(?:headlines?\s+for|news\s+about)\s+(.+)$'
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, lower_query)
            if match:
                clean_query = match.group(1).strip()
                break
        
        # Last resort - clean manually
        if not clean_query:
            clean_query = re.sub(r'^(?:fetch|find|get|give\s+me)\s+', '', lower_query)
            clean_query = re.sub(r'(?:top|latest)\s+\d+\s+', '', clean_query)
            clean_query = re.sub(r'(?:news|headlines?|articles?)\s*', '', clean_query)
            clean_query = clean_query.strip()
    
    return max_results, clean_query, content_type


def demo_news_fetcher():
    """
    Demonstration of how to use the news fetcher function.
    """
    
    # Example queries
    # test_queries = [
    #     "Fetch top 3 headlines for AAPL",
    #     "Get latest news about Elon Musk",
    #     "Give me top 5 trending headlines on AI technology",
    #     "Find latest 2 news about Bitcoin"
    # ]
    test_queries = [
        "find top 5 headlines on Nagpur"
    ]
    
    print("=== News Fetcher Demo ===\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("Response:")
        
        try:
            result = fetch_latest_news(query)
            print(result)
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    # Run demo
    demo_news_fetcher()