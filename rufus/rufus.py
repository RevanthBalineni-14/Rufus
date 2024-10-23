import atexit
import os
import logging
from .ner import extract_keywords_from_prompt
from .utils import expand_keywords_in_prompt, generate_cohesive_summary
from .crawler import crawl
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Configure logging for required information only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Rufus:
    def __init__(self, prompt, url, depth=4, max_pages=100):
        logging.info("Initializing Rufus with prompt and URL.")
        self.prompt = prompt
        self.url = url
        self.depth = depth
        self.max_pages = max_pages
        self.visited = set(["https://www.withchima.com/careers"])
        self.extracted_content = []

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.driver = self._initialize_driver()

        # Extract and log keywords
        self.keywords = self.extract_keywords()
        logging.info(f"Extracted Keywords: {self.keywords}")

        self.keywords_embedding = self.compute_keywords_embedding()

    def _initialize_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        try:
            driver = webdriver.Chrome(options=chrome_options)
            atexit.register(driver.quit)
            return driver
        except Exception as e:
            logging.error(f"Error initializing WebDriver: {e}")
            return None

    def extract_keywords(self):
        expanded_prompt = expand_keywords_in_prompt(self.prompt)
        return extract_keywords_from_prompt(expanded_prompt)

    def compute_keywords_embedding(self):
        return self.embedding_model.encode(' '.join(self.keywords), convert_to_tensor=True)

    def run(self):
        if not self.driver:
            logging.error("WebDriver not initialized. Aborting crawl.")
            return "Error: WebDriver not initialized."

        logging.info("Starting the crawl...")
        self.extracted_content = crawl(
            self.url,
            self.keywords_embedding,
            self.depth,
            self.driver,
            self.visited,
            self.max_pages
        )

        cleaned_text = ' '.join(self.extracted_content)
        summary = generate_cohesive_summary(self.prompt, cleaned_text)

        # Log the final summary
        # logging.info(f"Final Summary:\n{summary}")
        return summary
