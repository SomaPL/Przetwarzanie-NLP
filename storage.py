import json
import os

DATA_FILE = "sentences.json"

LABEL_ALIASES = {
    "pozytywny": "pozytywny",
    "neutralny": "neutralny",
    "negatywny": "negatywny",
    "naturalna": "neutralny",
    "neutralna": "neutralny",
    "positive": "pozytywny",
    "negative": "negatywny",
    "neutral": "neutralny",
}


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    return LABEL_ALIASES.get(label, label)


def ensure_data_file(path: str = DATA_FILE) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_sentences(path: str = DATA_FILE) -> list[dict]:
    ensure_data_file(path)

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                return []
        except json.JSONDecodeError:
            return []

    cleaned = []
    for item in data:
        if isinstance(item, dict) and "text" in item and "class" in item:
            cleaned.append({
                "text": str(item["text"]).strip(),
                "class": normalize_label(str(item["class"])),
            })

    return cleaned


def save_sentences(records: list[dict], path: str = DATA_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def append_sentence_record(text: str, label: str, path: str = DATA_FILE) -> None:
    records = load_sentences(path)
    records.append({
        "text": text.strip(),
        "class": normalize_label(label),
    })
    save_sentences(records, path)