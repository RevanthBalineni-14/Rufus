from nltk.tokenize import sent_tokenize
import nltk
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import trafilatura

nltk.download('punkt')

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

ABBREVIATION_MAP = {
    "HR": "Human Resources",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning"
}

def extract_main_content(html_content):
    """Extracts main readable content from HTML using Trafilatura."""
    return trafilatura.extract(html_content) or ''

def expand_keywords_in_prompt(prompt):
    """Expands common abbreviations in the prompt to improve keyword extraction."""
    words = prompt.split()
    return " ".join(ABBREVIATION_MAP.get(word, word) for word in words)

def split_text(text, max_chunk_size=1024):
    """Splits large text into smaller chunks suitable for the summarization model."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ''

    for sentence in sentences:
        if len(tokenizer.encode(current_chunk + ' ' + sentence)) < max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def clip_query_from_summary(summary):
    """
    Removes the part of the summary that contains the query or any sentence up to the first question mark.
    """
    # Check if the query or a question-like part exists in the summary
    if '?' in summary:
        # Split the summary at the first question mark and return the rest
        clipped_summary = summary.split('?', 1)[-1].strip()
        return clipped_summary
    return summary.strip()

def generate_cohesive_summary(query, text):
    """
    Generates a query-aware, cohesive summary from the extracted content.
    Properly truncates input to fit within the model's limits.
    """
    # Step 1: Split the text into manageable chunks
    max_input_length = 1024  # BART's maximum token length
    chunks = split_text(text, max_chunk_size=max_input_length)
    combined_summary = ""

    # Step 2: Generate summaries for each chunk, guided by the query
    for chunk in tqdm(chunks, desc="Generating Summary", unit="chunk"):
        # Combine query with the content for query-driven summarization
        input_text = f"Answer the question: {query}\nContent: {chunk}"

        # Truncate input to fit model limits
        inputs = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)

        try:
            # Generate query-specific summary with beam search for coherence
            summary_ids = model.generate(
                inputs,
                max_length=200,  # Length of each summary section
                min_length=50,   # Ensure detailed output
                num_beams=5,     # More beams for better coherence
                early_stopping=True
            )
            # Decode and add the summary to the final output
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            combined_summary += clip_query_from_summary(summary.strip()) + " "
        except Exception as e:
            print(f"Error generating summary: {e}. Skipping this chunk.")

    # Step 3: Return the cohesive, query-specific summary
    return combined_summary.strip()
