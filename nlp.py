import re
from collections import Counter
from pathlib import Path
from scipy.sparse import spmatrix

import requests
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nlp = spacy.load("pl_core_news_sm")
stemmer = SnowballStemmer("english")

STOPWORDS_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/stopwords-pl.txt"
STOPWORDS_PATH = Path("data/stopwords_pl.txt")
STOPWORDS_PATH.parent.mkdir(exist_ok=True)

if not STOPWORDS_PATH.exists():
    response = requests.get(STOPWORDS_URL, timeout=10)
    response.raise_for_status()
    STOPWORDS_PATH.write_text(response.text, encoding="utf-8")

with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    STOPWORDS = set(line.strip() for line in f if line.strip())


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\sąćęłńóśźż]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else [text.strip()]


def tokenize(text: str) -> list[str]:
    doc = nlp(text)
    return [token.text for token in doc if token.text.strip()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in STOPWORDS and t.isalpha()]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if token.text.strip()]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stemmer.stem(token) for token in tokens]


def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return list(ngrams(tokens, n))


def text_stats(tokens: list[str]) -> dict:
    counts = Counter(tokens)
    avg_len = round(sum(len(t) for t in tokens) / len(tokens), 2) if tokens else 0.0

    def class_counts_from_labels(labels: list[str]) -> dict:
        return dict(Counter(labels))

    return {
        "token_count": len(tokens),
        "unique_tokens": len(set(tokens)),
        "avg_token_length": avg_len,
        "top_words": counts.most_common(10),
        "unique_token_list": list(sorted(set(tokens)))[:100],
        "class_counts_from_labels": class_counts_from_labels,
    }


def bag_of_words(texts: list[str]) -> dict:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return {
        "features": vectorizer.get_feature_names_out().tolist(),
        "matrix": X.toarray().tolist(),
    }


def tfidf_features(texts: list[str]) -> dict:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return {
        "features": vectorizer.get_feature_names_out().tolist(),
        "matrix": X.toarray().round(3).tolist(),
    }