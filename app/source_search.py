import os
import requests
from typing import List


NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
BASE_URL = "https://newsapi.org/v2/everything"




def search_newsapi(query: str, page_size: int = 3) -> List[str]:
    """Return up to page_size article URLs that match the query."""
    if not NEWSAPI_KEY:
        return []
    params = {
    "q": query,
    "pageSize": page_size,
    "language": "en",
    "apiKey": NEWSAPI_KEY
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        urls = [a["url"] for a in data.get("articles", []) if a.get("url")]
        return urls
    except Exception:
        return []