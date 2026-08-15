from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
from dotenv import load_dotenv
from rich import print
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

@tool
def web_search(query : str) -> str:
    """Search the web for recent and reliable information on a topic . Returns Titles , URLs and snippets."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")
    if tavily is None:
        raise RuntimeError("Missing TAVILY_API_KEY. Add it to your .env file before running the pipeline.")
    try:
        results = tavily.search(query=query, max_results=5)
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed: {exc}") from exc

    out = []

    for r in results.get('results', []):
        out.append(
            f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('content', '')[:300]}\n"
        )
    
    return "\n----\n".join(out) if out else "No search results were returned for this query."

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
