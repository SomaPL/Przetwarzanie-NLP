from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LanguageResult:
    language: str
    confidence: float | None
    details: str

    def as_message(self) -> str:
        score = f"{self.confidence:.3f}" if self.confidence is not None else "brak"
        return f"Język: {self.language}\nPewność: {score}\nSzczegóły: {self.details}"


def detect_language(text: str) -> LanguageResult:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Tekst nie może być pusty.")

    try:
        from langdetect import detect_langs

        guesses = detect_langs(cleaned)
        if guesses:
            best = guesses[0]
            return LanguageResult(best.lang, float(best.prob), "langdetect")
    except Exception:
        pass

    return LanguageResult(simple_language_guess(cleaned), None, "prosty fallback znaków/słów")


def simple_language_guess(text: str) -> str:
    lower = text.lower()
    if any(ch in lower for ch in "ąćęłńóśźż") or any(
        word in lower.split() for word in ["jest", "oraz", "który", "która", "się"]
    ):
        return "pl"
    if any(word in lower.split() for word in ["the", "and", "is", "with", "this"]):
        return "en"
    if any(word in lower.split() for word in ["und", "ist", "der", "die", "das"]):
        return "de"
    if any(word in lower.split() for word in ["le", "la", "est", "avec", "des"]):
        return "fr"
    return "en"

