from __future__ import annotations

import csv
import time
from dataclasses import dataclass

import requests

from lab4_config import LAB4_SUMMARIES_PATH, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL, ensure_lab4_dirs
from lab4_language import detect_language


@dataclass
class SummaryResult:
    model: str
    text_length: int
    summary_type: str
    length: str
    summary: str
    seconds: float

    def as_message(self) -> str:
        return (
            f"Model: {self.model}\n"
            f"Text length: {self.text_length} tokens\n"
            f"Summary type: {self.summary_type}\n"
            f"Summary length: {self.length}\n\n"
            f"SUMMARY:\n{self.summary}\n\n"
            f"Generation time: {self.seconds:.2f}s"
        )


def summarize_text(
    text: str,
    summary_type: str = "abstractive",
    length: str = "medium",
    model: str = OLLAMA_MODEL,
    custom_prompt: str = "",
) -> SummaryResult:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Tekst nie może być pusty.")

    summary_type = summary_type.lower().strip()
    length = length.lower().strip()
    language = detect_language(cleaned).language
    prompt = build_prompt(cleaned, summary_type, length, language, custom_prompt)

    start = time.perf_counter()
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    seconds = time.perf_counter() - start
    summary = response.json().get("response", "").strip()
    if not summary:
        raise RuntimeError("Ollama zwróciła pustą odpowiedź.")

    result = SummaryResult(model, len(cleaned.split()), summary_type, length, summary, seconds)
    save_summary(cleaned, result)
    return result


def build_prompt(text: str, summary_type: str, length: str, language: str, custom_prompt: str) -> str:
    length_hint = {
        "short": "maksymalnie 3 krótkie zdania",
        "medium": "około 5-7 zdań",
        "long": "bardziej szczegółowo, 8-12 zdań",
    }.get(length, "około 5 zdań")

    type_hint = {
        "extractive": "Wybierz najważniejsze zdania z tekstu i nie dopisuj nowych informacji.",
        "abstractive": "Napisz własnymi słowami spójne streszczenie.",
        "bullets": "Zwróć streszczenie w punktach.",
    }.get(summary_type, "Napisz spójne streszczenie.")

    if custom_prompt:
        type_hint = custom_prompt

    return (
        f"Odpowiadaj w języku: {language}.\n"
        f"Zadanie: {type_hint}\n"
        f"Długość: {length_hint}.\n\n"
        f"Tekst:\n{text}\n\n"
        "Streszczenie:"
    )


def save_summary(source_text: str, result: SummaryResult) -> None:
    ensure_lab4_dirs()
    exists = LAB4_SUMMARIES_PATH.exists()
    with LAB4_SUMMARIES_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["model", "summary_type", "length", "text_length", "seconds", "source_text", "summary"])
        writer.writerow(
            [
                result.model,
                result.summary_type,
                result.length,
                result.text_length,
                round(result.seconds, 3),
                source_text,
                result.summary,
            ]
        )

