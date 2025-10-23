import os
import requests
from typing import List


BASE_URL = "https://newsdata.io/api/1/latest"


def search_newsapi(query: str, api_key: str, page_size: int = 3) -> List[str]:
    """Return up to page_size article URLs that match the query."""
    if not api_key:
        return []
    params = {
    "q": query,
    "pageSize": page_size,
    "language": "en",
    "apiKey": api_key
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        # The API uses the 'results' key for articles and 'link' for the URL.
        urls = [a["link"] for a in data.get("results", []) if a.get("link")]
        return urls
    except Exception as e:
        # It's good practice to log the error for debugging.
        print(f"NewsAPI search failed: {e}")
        return []