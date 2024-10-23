from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import requests
import nltk
from bs4 import BeautifulSoup
from transformers import pipeline
from tqdm import tqdm
nltk.download('punkt')
nltk.download('punkt_tab')

ABBREVIATION_MAP = {
    "HR": "Human Resources",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning"
}

def expand_keywords_in_prompt(prompt):
    words = prompt.split()
    expanded_words = [ABBREVIATION_MAP.get(word, word) for word in words]
    return " ".join(expanded_words)

def extract_relevant_sentences(text, keywords):
    sentences = sent_tokenize(text)
    return [s for s in sentences if any(k.lower() in s.lower() for k in keywords)]


def extract_relevant_links(page_html, keywords, base_url):
    soup = BeautifulSoup(page_html, 'html.parser')
    relevant_links = []

    for link in soup.find_all('a', href=True):
        anchor_text = link.get_text().strip()
        href = link['href']

        if any(keyword.lower() in anchor_text.lower() or keyword.lower() in href.lower() for keyword in keywords):
            full_url = requests.compat.urljoin(base_url, href)
            relevant_links.append(full_url)

    return relevant_links

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text_elements = soup.find_all(['p', 'h1', 'h2', 'li'])
    cleaned_text = "\n".join(element.get_text(separator=" ", strip=True) for element in text_elements)

    return cleaned_text


def split_text(text, max_chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

def summarize_text(text, max_length=150, min_length=50):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="tf")
    except Exception as e:
        print(f"Error initializing summarizer: {e}")
        return "Failed to initialize summarizer."

    chunks = split_text(text)
    summaries = []

    print(f"Summarizing {len(chunks)} chunks...")
    for chunk in tqdm(chunks, desc="Summarizing", unit="chunk"):
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")

    return " ".join(summaries)
