"""
Text preprocessing utilities.

This module contains common cleaning functions used by both
classical NLP models and Transformer-based models.
"""

from bs4 import BeautifulSoup


def remove_html(text: str) -> str:
    """
    Remove HTML tags from text.
    """

    if not isinstance(text, str):
        return ""

    if "<" not in text or ">" not in text:
        return text

    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


# sample = "<div>Hello <b>World</b></div>"

# print(remove_html(sample))

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def remove_urls(text: str) -> str:

    if not isinstance(text, str):
        return ""

    return URL_PATTERN.sub("", text)


# sample = "Visit https://amazon.com now!"

# print(remove_urls(sample))

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:

    if not isinstance(text, str):
        return ""

    return WHITESPACE_PATTERN.sub(" ", text).strip()


# text = "Hello      World\n\nAmazon"

# print(normalize_whitespace(text))


def to_lowercase(text: str) -> str:

    if not isinstance(text, str):
        return ""

    return text.lower()


def clean_text(text: str) -> str:
    """Common cleaning steps for all NLP pipelines."""

    text = remove_html(text)
    text = remove_urls(text)
    text = normalize_whitespace(text)

    return text


def preprocess_for_transformer(text: str) -> str:
    """
    Preprocessing pipeline for Transformer models.
    """

    return clean_text(text)


def preprocess_for_tfidf(text: str) -> str:
    """
    Preprocessing pipeline for classical ML models.
    """

    text = clean_text(text)
    text = to_lowercase(text)

    return text
