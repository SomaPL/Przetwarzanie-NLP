from __future__ import annotations

import csv
from dataclasses import dataclass

from lab4_config import LAB4_ENTITIES_PATH, ensure_lab4_dirs
from lab4_language import detect_language


SPACY_CACHE = {}
STANZA_CACHE = {}


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    method: str

    def as_line(self) -> str:
        return f"- {self.text} ({self.label}) [{self.start}:{self.end}]"


@dataclass
class NerResult:
    method: str
    text: str
    language: str
    entities: list[Entity]

    def as_message(self) -> str:
        lines = [
            f"Metoda: {self.method}",
            f"Język: {self.language}",
            f"TEXT: {self.text}",
            "",
            "ENTITIES:",
        ]
        if not self.entities:
            lines.append("- brak encji")
        else:
            lines.extend(entity.as_line() for entity in self.entities)
        return "\n".join(lines)


def run_ner(method: str, text: str, language: str | None = None) -> NerResult:
    method = method.lower().strip()
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Tekst nie może być pusty.")
    language = language or detect_language(cleaned).language

    if method == "spacy":
        result = ner_spacy(cleaned, language)
    elif method == "stanza":
        result = ner_stanza(cleaned, language)
    else:
        raise ValueError("Metoda NER musi mieć wartość spacy albo stanza.")

    save_entities(result)
    return result


def ner_spacy(text: str, language: str) -> NerResult:
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("Brakuje spaCy. Uruchom: python -m pip install -r requirements.txt") from exc

    model_name = "pl_core_news_sm" if language == "pl" else "en_core_web_sm"
    if model_name not in SPACY_CACHE:
        try:
            SPACY_CACHE[model_name] = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(f"Brakuje modelu spaCy: python -m spacy download {model_name}") from exc

    doc = SPACY_CACHE[model_name](text)
    entities = [
        Entity(ent.text, normalize_label(ent.label_), ent.start_char, ent.end_char, "spacy")
        for ent in doc.ents
    ]
    return NerResult("spacy", text, language, entities)


def ner_stanza(text: str, language: str) -> NerResult:
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError("Brakuje Stanza. Uruchom: python -m pip install -r requirements.txt") from exc

    if language not in STANZA_CACHE:
        try:
            STANZA_CACHE[language] = stanza.Pipeline(
                lang=language,
                processors="tokenize,ner",
                use_gpu=False,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Brakuje modelu Stanza. Uruchom: python -c \"import stanza; stanza.download('{language}')\""
            ) from exc

    doc = STANZA_CACHE[language](text)
    entities = [
        Entity(ent.text, normalize_label(ent.type), ent.start_char or 0, ent.end_char or 0, "stanza")
        for ent in doc.ents
    ]
    return NerResult("stanza", text, language, entities)


def normalize_label(label: str) -> str:
    mapping = {
        "PER": "PERSON",
        "persName": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORG",
        "orgName": "ORG",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "placeName": "LOCATION",
        "GPE": "GPE",
        "FAC": "FACILITY",
    }
    return mapping.get(label, label)


def save_entities(result: NerResult) -> None:
    ensure_lab4_dirs()
    exists = LAB4_ENTITIES_PATH.exists()
    with LAB4_ENTITIES_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["method", "language", "text", "entity", "label", "start", "end"])
        for entity in result.entities:
            writer.writerow(
                [result.method, result.language, result.text, entity.text, entity.label, entity.start, entity.end]
            )
