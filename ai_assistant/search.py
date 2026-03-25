
import requests
from googlesearch import search
from bs4 import BeautifulSoup
import os

class SearchEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def search_web(self, query: str, num_results: int = 3) -> str:
        """
        Performs a web search using the DeepSeek Search API if available,
        or falls back to a custom scraper/googlesearch-python.
        """
        if self.api_key:
            return self._search_deepseek(query)
        else:
            return self._search_fallback(query, num_results)

    def _search_deepseek(self, query: str) -> str:
        """Example implementation for DeepSeek's search API."""
        # DeepSeek Search API (Hypothetical endpoint/logic based on current trends)
        api_url = "https://api.deepseek.com/v1/search"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"query": query}
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=20)
            response.raise_for_status()
            results = response.json().get('results', [])
            return "\n".join([f"{r['title']}: {r['snippet']}" for r in results[:3]])
        except Exception as e:
            print(f"DeepSeek Search failed, falling back to scraper: {e}")
            return self._search_fallback(query)

    def _search_fallback(self, query: str, num_results: int = 3) -> str:
        """
        Custom fallback search using googlesearch-python and basic scraping.
        """
        print(f"--- Searching fallback for: {query} ---")
        try:
            results = []
            for url in search(query, num_results=num_results):
                try:
                    res = requests.get(url, timeout=5)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    # Get title and some text
                    title = soup.title.string if soup.title else url
                    text = ' '.join(soup.get_text().split())[:300] # Limit snippet
                    results.append(f"Source: {url}\nTitle: {title}\nSnippet: {text}\n")
                except Exception as inner_e:
                    results.append(f"Source: {url} (Content not accessible: {inner_e})")

            return "\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Search fallback error: {e}"
