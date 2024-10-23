import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords_from_prompt(prompt):
    doc = nlp(prompt)
    # print(doc.ents)
    keywords = set() 

    for ent in doc.ents:
        keywords.add(ent.text)

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            keywords.add(token.text)

    return list(keywords)
