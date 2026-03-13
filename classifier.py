from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from storage import load_sentences

LABEL_TO_ID = {
    "pozytywny": 1,
    "neutralny": 0,
    "negatywny": -1,
}

ID_TO_LABEL = {
    1: "pozytywny",
    0: "neutralny",
    -1: "negatywny",
}


def build_training_data() -> tuple[list[str], list[int]]:
    records = load_sentences()

    texts = []
    labels = []

    for record in records:
        text = record["text"].strip()
        label = record["class"].strip().lower()

        if text and label in LABEL_TO_ID:
            texts.append(text)
            labels.append(LABEL_TO_ID[label])

    return texts, labels


def train_model() -> Pipeline:
    texts, labels = build_training_data()

    if len(texts) < 3:
        raise ValueError("Za mało danych w sentences.json. Dodaj co najmniej 3 przykłady.")

    if len(set(labels)) < 2:
        raise ValueError("W danych muszą być co najmniej 2 różne klasy.")

    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(texts, labels)
    return model


def predict_text_class(text: str) -> str:
    model = train_model()
    pred = model.predict([text])[0]
    return ID_TO_LABEL[int(pred)]