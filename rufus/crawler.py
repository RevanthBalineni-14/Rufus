import requests
from bs4 import BeautifulSoup
from .utils import extract_relevant_sentences, extract_relevant_links

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    )
}

def crawl(url, keywords, depth, visited=set()):
    if depth == 0 or url in visited:
        return []
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=5, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

    visited.add(url)
    page_html = response.text
    relevant_sentences = extract_relevant_sentences(page_html, keywords)

    if depth > 1:
        relevant_links = extract_relevant_links(page_html, keywords, url)
        print(f"Found {len(relevant_links)} relevant links on {url}")
        for link in relevant_links:
            relevant_sentences += crawl(link, keywords, depth - 1, visited)

    return relevant_sentences
