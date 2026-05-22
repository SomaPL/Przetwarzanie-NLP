from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sentiment_config import SENTIMENT_LABELS
from sentiment_datasets import load_sentiment_dataset


@dataclass
class SentimentPrediction:
    method: str
    label: str
    score: float | None
    details: str
    model_path: str = ""

    def as_message(self) -> str:
        lines = [
            f"Metoda: {self.method}",
            f"Predykcja: {self.label}",
        ]
        if self.score is not None:
            lines.append(f"Pewność/ocena: {self.score:.3f}")
        if self.model_path:
            lines.append(f"Model: {self.model_path}")
        if self.details:
            lines.append(f"Szczegóły: {self.details}")
        return "\n".join(lines)


POSITIVE_WORDS = {
    "świetny",
    "swietny",
    "dobry",
    "wspaniały",
    "wspanialy",
    "super",
    "polecam",
    "zadowolony",
    "zadowolona",
    "uwielbiam",
    "najlepszy",
    "znakomity",
    "pozytywnie",
    "szybko",
    "pomocna",
    "miła",
    "mila",
}

NEGATIVE_WORDS = {
    "zły",
    "zly",
    "fatalny",
    "okropny",
    "nudny",
    "rozczarowany",
    "rozczarowana",
    "uszkodzony",
    "zepsuty",
    "wolno",
    "frustrujący",
    "frustrujacy",
    "nie polecam",
    "strata",
    "słaba",
    "slaba",
    "opóźniona",
    "opozniona",
}


def predict_sentiment(method: str, text: str, dataset: str = "custom") -> SentimentPrediction:
    method = method.lower().strip()
    text = text.strip()
    if not text:
        raise ValueError("Tekst nie może być pusty.")

    if method == "rule":
        return predict_rule(text)
    if method in {"nb", "rf"}:
        return predict_classical_ml(method, text, dataset)
    if method == "transformer":
        return predict_transformer(text)
    if method == "textblob":
        return predict_textblob(text)
    if method == "stanza":
        return predict_stanza(text)
    if method in {"simplernn", "lstm", "gru"}:
        from sentiment_training import predict_with_sequence_model

        return predict_with_sequence_model(method, dataset, text)

    raise ValueError("Nieznana metoda sentymentu.")


def predict_rule(text: str) -> SentimentPrediction:
    lower = text.lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in lower)
    negative = sum(1 for word in NEGATIVE_WORDS if word in lower)
    score = positive - negative

    if score > 0:
        label = "pozytywny"
    elif score < 0:
        label = "negatywny"
    else:
        label = "neutralny"

    confidence = min(1.0, abs(score) / 3) if score else 0.5
    details = f"słowa pozytywne={positive}, negatywne={negative}"
    return SentimentPrediction("rule", label, confidence, details)


def predict_classical_ml(method: str, text: str, dataset: str) -> SentimentPrediction:
    df = load_sentiment_dataset(dataset)
    model = build_classical_pipeline(method)
    model.fit(df["text"], df["label"])
    label = str(model.predict([text])[0])

    score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        score = float(np.max(probabilities))

    details = f"model trenowany na dataset={dataset}, liczba rekordów={len(df)}"
    return SentimentPrediction(method, label, score, details)


def build_classical_pipeline(method: str) -> Pipeline:
    if method == "nb":
        classifier = MultinomialNB()
    elif method == "rf":
        classifier = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1)
    else:
        raise ValueError("Klasyczny model musi mieć wartość nb albo rf.")

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", classifier),
        ]
    )


def predict_transformer(text: str) -> SentimentPrediction:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Brakuje biblioteki transformers. Uruchom: pip install -r requirements.txt") from exc

    try:
        classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None,
        )
        output = classifier(text)
        result = output[0] if output and isinstance(output[0], list) else output
    except Exception as exc:
        raise RuntimeError(
            "Nie udało się uruchomić transformera. Przy pierwszym użyciu model musi zostać pobrany z internetu."
        ) from exc

    best = max(result, key=lambda item: item["score"])
    label = map_transformer_label(str(best["label"]))
    return SentimentPrediction(
        "transformer",
        label,
        float(best["score"]),
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    )


def map_transformer_label(label: str) -> str:
    normalized = label.lower()
    if "negative" in normalized or normalized in {"label_0", "0"}:
        return "negatywny"
    if "neutral" in normalized or normalized in {"label_1", "1"}:
        return "neutralny"
    if "positive" in normalized or normalized in {"label_2", "2"}:
        return "pozytywny"
    if "1 star" in normalized or "2 star" in normalized:
        return "negatywny"
    if "3 star" in normalized:
        return "neutralny"
    if "4 star" in normalized or "5 star" in normalized:
        return "pozytywny"
    return normalized


def predict_textblob(text: str) -> SentimentPrediction:
    try:
        from textblob import TextBlob
    except ImportError as exc:
        raise RuntimeError("Brakuje biblioteki textblob. Uruchom: pip install -r requirements.txt") from exc

    polarity = float(TextBlob(text).sentiment.polarity)
    if polarity > 0.1:
        label = "pozytywny"
    elif polarity < -0.1:
        label = "negatywny"
    else:
        label = "neutralny"
    return SentimentPrediction("textblob", label, abs(polarity), f"polarity={polarity:.3f}")


def predict_stanza(text: str) -> SentimentPrediction:
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError("Brakuje biblioteki stanza. Uruchom: pip install -r requirements.txt") from exc

    try:
        nlp = stanza.Pipeline("pl", processors="tokenize,lemma", use_gpu=False, verbose=False)
        doc = nlp(text)
    except Exception as exc:
        raise RuntimeError(
            "Nie udało się uruchomić Stanza. Pobierz model języka: "
            "python -c \"import stanza; stanza.download('pl')\""
        ) from exc

    lemmas = [word.lemma.lower() for sentence in doc.sentences for word in sentence.words if word.lemma]
    joined = " ".join(lemmas)
    positive = sum(1 for word in POSITIVE_WORDS if word in joined)
    negative = sum(1 for word in NEGATIVE_WORDS if word in joined)
    score = positive - negative

    if score > 0:
        label = "pozytywny"
    elif score < 0:
        label = "negatywny"
    else:
        label = "neutralny"

    confidence = min(1.0, abs(score) / 3) if score else 0.5
    return SentimentPrediction("stanza", label, confidence, f"analiza lematów: {', '.join(lemmas[:12])}")


def ordered_labels(labels: list[str]) -> list[str]:
    ordered = [label for label in SENTIMENT_LABELS if label in labels]
    ordered.extend(sorted(label for label in set(labels) if label not in ordered))
    return ordered
