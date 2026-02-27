"""
Brave Search API wrapper for finding alpha ideas.

Searches for published alpha strategies, quant blog posts,
and academic papers related to crypto futures trading.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from src.openclaw.config import DEFAULT_SEARCH_THEMES, RESEARCH_POLICY

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass
class AlphaIdea:
    """Raw alpha idea from search results."""

    title: str
    url: str
    snippet: str
    source_type: str = "other"       # paper | blog | forum | other
    relevance_score: float = 0.0     # 0-1, set by LLM scoring
    raw_text: str = ""               # full fetched content (if fetched)


class BraveSearchClient:
    """
    Brave Web Search API wrapper for alpha idea discovery.

    Searches multiple themes and deduplicates results.
    """

    def __init__(
        self,
        api_key: str | None = None,
        results_per_query: int | None = None,
    ):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Brave API key required. Set BRAVE_API_KEY env var "
                "or pass api_key parameter."
            )
        self.results_per_query = (
            results_per_query or RESEARCH_POLICY.brave_results_per_query
        )
        self._seen_urls: set[str] = set()

    def search_alpha_ideas(
        self,
        query: str | None = None,
        query_themes: list[str] | None = None,
    ) -> list[AlphaIdea]:
        """
        Search for alpha ideas using Brave Web Search.

        Args:
            query: Single search query (used as-is if provided)
            query_themes: Multiple search themes (default: DEFAULT_SEARCH_THEMES)

        Returns:
            Deduplicated list of AlphaIdea objects.
        """
        ideas: list[AlphaIdea] = []

        if query:
            # Single query mode
            results = self._search(query)
            ideas.extend(results)
        else:
            # Multi-theme mode
            themes = query_themes or DEFAULT_SEARCH_THEMES
            for theme in themes:
                results = self._search(theme)
                ideas.extend(results)

                # Stop if we have enough ideas
                if len(ideas) >= self.results_per_query * 2:
                    break

        # Deduplicate by URL
        unique_ideas = []
        for idea in ideas:
            if idea.url not in self._seen_urls:
                self._seen_urls.add(idea.url)
                unique_ideas.append(idea)

        logger.info(
            f"Brave search: {len(unique_ideas)} unique ideas "
            f"from {len(ideas)} total results"
        )

        return unique_ideas

    def _search(self, query: str) -> list[AlphaIdea]:
        """Execute a single Brave search query."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": self.results_per_query,
            "search_lang": "en",
            "freshness": "py",  # past year
        }

        try:
            resp = requests.get(
                BRAVE_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Brave search failed for '{query}': {e}")
            return []

        ideas = []
        web_results = data.get("web", {}).get("results", [])

        for result in web_results:
            title = result.get("title", "")
            url = result.get("url", "")
            snippet = result.get("description", "")

            if not url or not title:
                continue

            source_type = self._classify_source(url, title)

            ideas.append(AlphaIdea(
                title=title,
                url=url,
                snippet=snippet,
                source_type=source_type,
            ))

        logger.debug(f"Query '{query[:50]}': {len(ideas)} results")
        return ideas

    @staticmethod
    def _classify_source(url: str, title: str) -> str:
        """Classify the source type based on URL and title."""
        url_lower = url.lower()
        title_lower = title.lower()

        if any(d in url_lower for d in [
            "arxiv.org", "ssrn.com", "nber.org", "sciencedirect.com",
            "springer.com", "ieee.org", "acm.org",
        ]):
            return "paper"

        if any(d in url_lower for d in [
            "medium.com", "substack.com", "towardsdatascience.com",
            "quantpedia.com", "alphaarchitect.com", "blog",
        ]):
            return "blog"

        if any(d in url_lower for d in [
            "reddit.com", "quantconnect.com/forum", "stackoverflow.com",
            "github.com", "kaggle.com",
        ]):
            return "forum"

        return "other"

    def fetch_content(self, url: str, max_chars: int = 10000) -> str:
        """
        Fetch and extract main text from a URL.

        Returns truncated text content for LLM processing.
        """
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "OpenClaw Alpha Researcher/1.0"},
            )
            resp.raise_for_status()

            # Simple HTML text extraction (strip tags)
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts: list[str] = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "header", "footer"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "header", "footer"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)

            extractor = TextExtractor()
            extractor.feed(resp.text)
            full_text = " ".join(extractor.text_parts)

            return full_text[:max_chars]

        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return ""
