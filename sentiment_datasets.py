from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_config import CUSTOM_DATASET_PATH, DATA_DIR, SENTIMENT_LABELS


SAMPLE_ROWS = [
    ("Uwielbiam ten film, był naprawdę świetny.", "pozytywny"),
    ("Produkt działa znakomicie i spełnił moje oczekiwania.", "pozytywny"),
    ("Obsługa była bardzo miła i pomocna.", "pozytywny"),
    ("To była jedna z najlepszych decyzji zakupowych.", "pozytywny"),
    ("Aplikacja działa szybko i wygodnie.", "pozytywny"),
    ("Jestem bardzo zadowolony z jakości.", "pozytywny"),
    ("Film miał świetną muzykę i dobrą grę aktorską.", "pozytywny"),
    ("Dostawa przyszła szybko, wszystko było w porządku.", "pozytywny"),
    ("Polecam, naprawdę warto spróbować.", "pozytywny"),
    ("Ten produkt pozytywnie mnie zaskoczył.", "pozytywny"),
    ("To był zwykły dzień bez większych emocji.", "neutralny"),
    ("Produkt jest poprawny, ale niczym szczególnym się nie wyróżnia.", "neutralny"),
    ("Film był przeciętny i raczej przewidywalny.", "neutralny"),
    ("Obsługa była poprawna, ale bez większego zaangażowania.", "neutralny"),
    ("Aplikacja ma podstawowe funkcje i działa normalnie.", "neutralny"),
    ("Nie mam mocnej opinii na ten temat.", "neutralny"),
    ("Dostawa była zgodna z podanym terminem.", "neutralny"),
    ("Jakość jest adekwatna do ceny.", "neutralny"),
    ("To rozwiązanie jest wystarczające do prostych zadań.", "neutralny"),
    ("Wszystko przebiegło standardowo.", "neutralny"),
    ("Ten produkt jest fatalny i szybko się zepsuł.", "negatywny"),
    ("Film był nudny, za długi i rozczarowujący.", "negatywny"),
    ("Obsługa była niemiła i nie rozwiązała problemu.", "negatywny"),
    ("Jestem bardzo rozczarowany zakupem.", "negatywny"),
    ("Aplikacja ciągle się zawiesza i działa wolno.", "negatywny"),
    ("To była strata czasu i pieniędzy.", "negatywny"),
    ("Dostawa była opóźniona, a paczka uszkodzona.", "negatywny"),
    ("Jakość wykonania jest bardzo słaba.", "negatywny"),
    ("Nie polecam, doświadczenie było okropne.", "negatywny"),
    ("Kontakt ze wsparciem był frustrujący.", "negatywny"),
]


def ensure_custom_dataset() -> None:
    if CUSTOM_DATASET_PATH.exists():
        return

    with CUSTOM_DATASET_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(SAMPLE_ROWS)


def add_custom_record(text: str, label: str) -> int:
    ensure_custom_dataset()
    normalized_label = normalize_sentiment_label(label)
    if normalized_label not in SENTIMENT_LABELS:
        raise ValueError(f"Etykieta musi być jedną z: {', '.join(SENTIMENT_LABELS)}")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Tekst nie może być pusty.")

    with CUSTOM_DATASET_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([cleaned_text, normalized_label])

    return len(load_sentiment_dataset("custom"))


def load_sentiment_dataset(dataset: str) -> pd.DataFrame:
    name = dataset.lower().strip()
    if name == "custom":
        ensure_custom_dataset()
        df = _read_csv(CUSTOM_DATASET_PATH)
    elif name == "imdb":
        df = _load_external_csv(
            [
                DATA_DIR / "imdb.csv",
                DATA_DIR / "IMDB Dataset.csv",
                DATA_DIR / "imdb-dataset.csv",
            ]
        )
    elif name == "amazon":
        df = _load_external_csv(
            [
                DATA_DIR / "amazon.csv",
                DATA_DIR / "amazon_reviews.csv",
                DATA_DIR / "Reviews.csv",
            ]
        )
    else:
        raise ValueError("Dataset musi mieć wartość: amazon, imdb albo custom.")

    prepared = prepare_sentiment_frame(df)
    if prepared.empty:
        raise ValueError("Dataset jest pusty po oczyszczeniu danych.")
    if prepared["label"].nunique() < 2:
        raise ValueError("Dataset musi zawierać co najmniej dwie klasy sentymentu.")
    return prepared


def prepare_sentiment_frame(df: pd.DataFrame) -> pd.DataFrame:
    text_col = find_column(df, ["text", "review", "content", "comment", "sentence", "description"])
    label_col = find_column(df, ["label", "sentiment", "class", "target", "rating", "score", "overall"])
    if text_col is None or label_col is None:
        raise ValueError("Dataset powinien mieć kolumny tekstu i etykiety, np. text,label.")

    rows = df[[text_col, label_col]].dropna().copy()
    rows.columns = ["text", "label"]
    rows["text"] = rows["text"].astype(str).str.strip()
    rows["label"] = [normalize_sentiment_label(value, label_col) for value in rows["label"]]
    rows = rows[(rows["text"] != "") & (rows["label"] != "")]
    return rows.reset_index(drop=True)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {str(col).lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for col in df.columns:
        lower = str(col).lower()
        if any(candidate in lower for candidate in candidates):
            return col
    return None


def normalize_sentiment_label(value: Any, column_name: str = "label") -> str:
    if pd.isna(value):
        return ""

    raw = str(value).strip().lower()
    aliases = {
        "pozytywny": "pozytywny",
        "positive": "pozytywny",
        "pos": "pozytywny",
        "1": "pozytywny",
        "neutralny": "neutralny",
        "neutral": "neutralny",
        "neu": "neutralny",
        "0": "neutralny",
        "negatywny": "negatywny",
        "negative": "negatywny",
        "neg": "negatywny",
        "-1": "negatywny",
    }

    rating_columns = {"rating", "score", "overall", "stars"}
    if str(column_name).lower().strip() in rating_columns:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return aliases.get(raw, raw)
        if number <= 2:
            return "negatywny"
        if number >= 4:
            return "pozytywny"
        return "neutralny"

    return aliases.get(raw, raw)


def _load_external_csv(paths: list[Path]) -> pd.DataFrame:
    existing = next((path for path in paths if path.exists()), None)
    if existing is None:
        expected = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"Nie znaleziono pliku datasetu. Dodaj jeden z: {expected}")
    return _read_csv(existing)


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

