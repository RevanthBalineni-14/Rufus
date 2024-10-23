from setuptools import setup, find_packages

setup(
    name="rufus",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "spacy",
        "nltk"
    ],
)
