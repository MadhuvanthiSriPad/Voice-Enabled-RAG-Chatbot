import argparse
import logging
import os
import re
import sys
from typing import Optional, Protocol, Tuple

import requests
from bs4 import BeautifulSoup

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADING_SEPARATOR = "\n\n" + "=" * 20 + "\n{}\n" + "=" * 20 + "\n"


class SearchProvider(Protocol):
    """Protocol for search providers - defines the expected interface."""
    def search(self, query: str) -> Optional[Tuple[str, str]]: ...


class WikipediaAPISearch:
    def __init__(self, session: requests.Session):
        self.session = session

    def search(self, query: str) -> Optional[Tuple[str, str]]:
        result = self._opensearch(query)
        if result:
            return result

        corrected = self._get_spelling_suggestion(query)
        if corrected and corrected.lower() != query.lower():
            logger.info(f"Spelling corrected: '{query}' -> '{corrected}'")
            result = self._opensearch(corrected)
            if result:
                return result

        return self._fuzzy_search(query)

    def _opensearch(self, query: str) -> Optional[Tuple[str, str]]:
        try:
            params = {"action": "opensearch", "search": query, "limit": 5, "format": "json"}
            data = self.session.get(WIKIPEDIA_API, params=params, timeout=10).json()
            if len(data) >= 4 and data[1]:
                logger.info(f"Found via OpenSearch: {data[1][0]}")
                return data[1][0], data[3][0]
        except Exception as e:
            logger.error(f"OpenSearch failed: {e}")
        return None

    def _get_spelling_suggestion(self, query: str) -> Optional[str]:
        try:
            params = {"action": "query", "list": "search", "srsearch": query, "srinfo": "suggestion", "format": "json"}
            data = self.session.get(WIKIPEDIA_API, params=params, timeout=10).json()
            return data.get("query", {}).get("searchinfo", {}).get("suggestion")
        except Exception as e:
            logger.error(f"Spelling suggestion failed: {e}")
        return None

    def _fuzzy_search(self, query: str) -> Optional[Tuple[str, str]]:
        try:
            params = {"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json"}
            results = self.session.get(WIKIPEDIA_API, params=params, timeout=10).json().get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                logger.info(f"Found via fuzzy search: {title}")
                return title, f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
        return None


class WikipediaScraper:
    """Scrapes and cleans Wikipedia article content."""

    ELEMENTS_TO_REMOVE = [
        ('div', {'class': 'navbox'}), ('div', {'class': 'vertical-navbox'}),
        ('div', {'class': 'sidebar'}), ('div', {'class': 'infobox'}),
        ('div', {'class': 'toc'}), ('table', {'class': 'infobox'}),
        ('table', {'class': 'wikitable'}), ('sup', {'class': 'reference'}),
        ('span', {'class': 'mw-editsection'}), ('div', {'class': 'reflist'}),
        ('div', {'class': 'refbegin'}), ('div', {'class': 'thumb'}),
        ('div', {'class': 'hatnote'}), ('div', {'class': 'metadata'}),
        ('div', {'role': 'navigation'}), ('span', {'class': 'noprint'}),
        ('style', {}), ('script', {}),
    ]
    SECTIONS_TO_REMOVE = ['References', 'External_links', 'See_also', 'Notes', 'Further_reading']

    def __init__(self, session: requests.Session):
        self.session = session

    def scrape(self, url: str, title: str) -> Optional[str]:
        if not title:
            title = url.split('/wiki/')[-1].replace('_', ' ')
        return self._scrape_via_api(title) or self._scrape_via_html(url)

    def _scrape_via_api(self, title: str) -> Optional[str]:
        try:
            params = {"action": "query", "titles": title, "prop": "extracts", "explaintext": True, "format": "json"}
            pages = self.session.get(WIKIPEDIA_API, params=params, timeout=10).json().get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    return page_data.get("extract", "")
        except Exception as e:
            logger.error(f"API scrape failed: {e}")
        return None

    def _scrape_via_html(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return self._extract_text(response.text)
        except Exception as e:
            logger.error(f"HTML scrape failed: {e}")
        return None

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.find('div', {'id': 'mw-content-text'}) or soup.find('div', {'class': 'mw-parser-output'}) or soup.body

        self._remove_elements(content)
        self._remove_sections(content)

        text_parts = []
        for el in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
            text = el.get_text(strip=True)
            if text:
                text_parts.append(HEADING_SEPARATOR.format(text) if el.name.startswith('h') else text)

        return self._clean_text('\n'.join(text_parts))

    def _remove_elements(self, content) -> None:
        for tag, attrs in self.ELEMENTS_TO_REMOVE:
            for el in content.find_all(tag, attrs):
                el.decompose()

    def _remove_sections(self, content) -> None:
        for section_id in self.SECTIONS_TO_REMOVE:
            heading = content.find(id=section_id)
            if heading:
                parent = heading.find_parent(['h2', 'h3'])
                if parent:
                    for sibling in list(parent.find_next_siblings()):
                        if sibling.name in ['h2', 'h3']:
                            break
                        sibling.decompose()
                    parent.decompose()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


def write_file(content: str, filepath: str) -> bool:
    try:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved to: {filepath}")
        return True
    except IOError as e:
        logger.error(f"Write failed: {e}")
        return False


def collect(topic: str, output_path: str, search_provider: SearchProvider, scraper: WikipediaScraper) -> bool:
    print(f"\n[1] Searching for: '{topic}'")
    result = search_provider.search(topic)
    if not result:
        print(f"No Wikipedia article found for: '{topic}'")
        return False

    title, url = result
    if topic.lower() not in title.lower():
        print("    (Closest match found)")
    print(f"[2] Found: {title}\n    URL: {url}")

    print("[3] Scraping content...")
    content = scraper.scrape(url, title)
    if not content:
        print("Failed to scrape article")
        return False

    print(f"[4] Extracted {len(content)} characters")

    if write_file(content, output_path):
        print(f"\nSaved to: {output_path}")
        print(f"\nPreview:\n{content[:300]}...")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(prog='task1_data_collection')
    parser.add_argument('--topic', '-t', type=str, help='Search topic')
    parser.add_argument('--output', '-o', type=str, default=os.path.join(config.OUTPUT_DIR, config.SCRAPED_TEXT_FILE))
    args = parser.parse_args()

    if not args.topic:
        args.topic = input("Enter search topic: ").strip()
        if not args.topic:
            print("Error: Topic cannot be empty")
            return 1

    session = requests.Session()
    session.headers.update({"User-Agent": "VoiceRAGChatbot/1.0 (Educational Project)"})

    search_provider = WikipediaAPISearch(session)
    scraper = WikipediaScraper(session)

    return 0 if collect(args.topic, args.output, search_provider, scraper) else 1


if __name__ == "__main__":
    sys.exit(main())
