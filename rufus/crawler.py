import requests
from bs4 import BeautifulSoup
from .utils import extract_main_content
import heapq
from sentence_transformers import SentenceTransformer, util
import logging
from urllib.parse import urljoin, urlparse

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_relevance(text, keywords_embedding):
    """Compute cosine similarity between page content and keywords."""
    if not text.strip():
        return 0
    text_embedding = embedding_model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(keywords_embedding, text_embedding)
    return cosine_scores.item()

def fetch_page_content(url, driver, timeout=2):
    """
    Fetches the page content using Selenium, with timeout handling.
    """
    try:
        # Set timeouts to avoid long waits
        driver.set_page_load_timeout(timeout)  
        driver.set_script_timeout(timeout)   

        logging.info(f"Fetching content from: {url}")
        driver.get(url)  
        return driver.page_source

    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return ''

def crawl(url, keywords_embedding, depth, driver, visited, max_pages=100, relevance_threshold=0.2):
    """Recursively crawl URLs based on content relevance."""
    # Normalize the URL to ensure consistent tracking
    if url in visited:
        return []

    if depth == 0 or len(visited) >= max_pages:
        logging.info(f"Stopping at URL: {url} (Depth: {depth}, Visited: {len(visited)})")
        return []

    logging.info(f"Crawling URL: {url}")
    visited.add(url)
    page_html = fetch_page_content(url, driver)

    if not page_html:
        logging.warning(f"No content fetched from {url}.")
        return []

    # Extract the main content from the HTML
    page_text = extract_main_content(page_html)
    relevance_score = compute_relevance(page_text, keywords_embedding)
    logging.info(f"Relevance score for {url}: {relevance_score}")

    if relevance_score < relevance_threshold:
        logging.info(f"Content at {url} is not relevant. Stopping traversal.")
        return []

    # If the content is relevant, add it to the extracted content
    extracted_content = [page_text]

    # Collect links for further traversal
    links = []
    soup = BeautifulSoup(page_html, 'html.parser')
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        full_url = urljoin(url, href)

        if full_url not in visited:
            logging.info(f"Discovered new link: {full_url}")
            heapq.heappush(links, (0, full_url))  # Priority is not based on relevance anymore

    # Recursively crawl the collected links
    while links and len(visited) < max_pages:
        _, next_url = heapq.heappop(links)
        extracted_content += crawl(next_url, keywords_embedding, depth - 1, driver, visited, max_pages, relevance_threshold)

    return extracted_content
