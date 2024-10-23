import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rufus.ner import extract_keywords_from_prompt
from rufus.utils import expand_keywords_in_prompt , clean_html, summarize_text
from rufus.crawler import crawl

prompt = "We're making a chatbot for the HR in San Francisco."
keywords = expand_keywords_in_prompt(prompt)
print("Expanded Keywords:", keywords)

keywords = extract_keywords_from_prompt(''.join(keywords))
print("Extracted Keywords:", keywords)

url = "https://www.sf.gov/"
html_content = crawl(url, keywords, depth=4)
# print("Extracted Content:", content)

cleaned_text = clean_html(" ".join(html_content))
print("Cleaned Text:", cleaned_text)

summary = summarize_text(cleaned_text)
print("Summarized Content:", summary)
