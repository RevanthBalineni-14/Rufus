from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords_from_prompt(prompt):
    keywords = kw_model.extract_keywords(
        prompt, keyphrase_ngram_range=(1, 3), stop_words='english'
    )
    return [kw[0] for kw in keywords]
