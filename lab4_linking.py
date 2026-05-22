from __future__ import annotations

import math
import csv
import re
from dataclasses import dataclass
from urllib.parse import quote

import requests

from lab4_config import LAB4_LINKS_PATH, ensure_lab4_dirs


WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
WIKIPEDIA_SEARCH_URL = "https://{language}.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "Przetwarzanie-NLP-Lab4/1.0 (student project; Telegram NLP bot)"
}


@dataclass
class EntityCandidate:
    label: str
    wikidata_id: str
    description: str
    wikipedia_url: str
    confidence: float

    def as_lines(self, index: int) -> list[str]:
        return [
            f"{index}. {self.label} ({self.wikidata_id}) - {self.description or 'brak opisu'}",
            f"   - Wikipedia: {self.wikipedia_url or 'brak'}",
            f"   - Confidence: {self.confidence:.2f}",
        ]


def link_entity(entity: str, language: str = "en", context: str = "") -> list[EntityCandidate]:
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": language,
        "uselang": language,
        "search": entity,
        "limit": 6,
    }
    response = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    items = response.json().get("search", [])

    candidates: list[EntityCandidate] = []
    for rank, item in enumerate(items):
        wikidata_id = item.get("id", "")
        label = item.get("label", "")
        description = item.get("description", "")
        wikipedia_url = get_wikipedia_url(wikidata_id, language)
        confidence = score_candidate(entity, context, label, description, rank)
        candidates.append(
            EntityCandidate(label, wikidata_id, description, wikipedia_url, confidence)
        )

    if not candidates:
        candidates.extend(wikipedia_fallback(entity, language))

    return sorted(candidates, key=lambda item: item.confidence, reverse=True)


def disambiguate_entity(entity: str, context: str, language: str = "en") -> EntityCandidate | None:
    candidates = link_entity(entity, language, context)
    if not candidates:
        return None
    best = candidates[0]
    return best if best.confidence >= 0.15 else None


def save_link_candidates(entity: str, language: str, context: str, candidates: list[EntityCandidate]) -> None:
    ensure_lab4_dirs()
    exists = LAB4_LINKS_PATH.exists()
    with LAB4_LINKS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                ["entity", "language", "context", "rank", "label", "wikidata_id", "description", "wikipedia_url", "confidence"]
            )
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow(
                [
                    entity,
                    language,
                    context,
                    rank,
                    candidate.label,
                    candidate.wikidata_id,
                    candidate.description,
                    candidate.wikipedia_url,
                    round(candidate.confidence, 4),
                ]
            )


def get_wikipedia_url(wikidata_id: str, language: str) -> str:
    if not wikidata_id:
        return ""
    try:
        response = requests.get(
            WIKIDATA_ENTITY_URL.format(entity_id=wikidata_id),
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        entity = response.json()["entities"][wikidata_id]
        sitelink = entity.get("sitelinks", {}).get(f"{language}wiki")
        if not sitelink and language != "en":
            sitelink = entity.get("sitelinks", {}).get("enwiki")
        if not sitelink:
            return ""
        title = sitelink["title"]
        wiki_language = language if entity.get("sitelinks", {}).get(f"{language}wiki") else "en"
        return f"https://{wiki_language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    except Exception:
        return ""


def wikipedia_fallback(entity: str, language: str) -> list[EntityCandidate]:
    params = {
        "action": "opensearch",
        "search": entity,
        "limit": 5,
        "namespace": 0,
        "format": "json",
    }
    try:
        response = requests.get(
            WIKIPEDIA_SEARCH_URL.format(language=language),
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    labels = data[1] if len(data) > 1 else []
    descriptions = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []
    candidates = []
    for rank, label in enumerate(labels):
        candidates.append(
            EntityCandidate(
                label=label,
                wikidata_id="Wikipedia",
                description=descriptions[rank] if rank < len(descriptions) else "",
                wikipedia_url=urls[rank] if rank < len(urls) else "",
                confidence=max(0.1, 0.65 - rank * 0.1),
            )
        )
    return candidates


def score_candidate(entity: str, context: str, label: str, description: str, rank: int) -> float:
    entity_lower = entity.lower().strip()
    label_lower = label.lower().strip()
    base = 0.75 if entity_lower == label_lower else 0.45
    base -= rank * 0.08

    context_words = tokenize(context)
    description_words = tokenize(description)
    if context_words and description_words:
        overlap = len(context_words & description_words) / math.sqrt(len(context_words) * len(description_words))
        base += min(0.25, overlap)

    return max(0.01, min(0.99, base))


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"(?u)\b\w\w+\b", text.lower())
        if token not in {"oraz", "jest", "the", "and", "for", "with", "a", "an", "to"}
    }
