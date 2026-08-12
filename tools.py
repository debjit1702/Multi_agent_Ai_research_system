from langchain.tools import tool
import requests
from tavily import TavilyClient
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich import print

load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def _web_search_impl(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs and snippets."""

    results = tavily.search(
        query=query,
        max_results=5
    )

    out = []

    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content'][:300]}\n"
        )

    return "\n-----\n".join(out)

# LangChain expects a tool object; keep a wrapped version for use with LangChain
web_search = tool(_web_search_impl)

def _scrape_url_impl(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

# Wrapped tool for LangChain usage
scrape_url = tool(_scrape_url_impl)


if __name__ == "__main__":
    # call the plain implementation (not the StructuredTool wrapper)
    print(_scrape_url_impl("https://www.nbcnews.com/world/iran-war"))