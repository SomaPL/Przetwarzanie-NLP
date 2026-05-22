from __future__ import annotations

import csv
from dataclasses import dataclass

from lab4_config import LAB4_TRANSLATIONS_PATH, TRANSLATION_MODELS, TRANSLATION_PREFIXES, ensure_lab4_dirs
from lab4_language import detect_language


TRANSLATOR_CACHE = {}


@dataclass
class TranslationResult:
    source_lang: str
    target_lang: str
    translation: str
    models: list[str]

    def as_message(self) -> str:
        return (
            f"Source: {self.source_lang}\n"
            f"Target: {self.target_lang}\n"
            f"Modele: {', '.join(self.models)}\n\n"
            f"Translation:\n{self.translation}"
        )


def translate_text(text: str, target_lang: str, source_lang: str | None = None) -> TranslationResult:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Tekst nie może być pusty.")

    source_lang = source_lang or detect_language(cleaned).language
    target_lang = target_lang.lower().strip()
    if source_lang == target_lang:
        return TranslationResult(source_lang, target_lang, cleaned, ["bez tłumaczenia"])

    path = translation_path(source_lang, target_lang)
    current_text = cleaned
    used_models: list[str] = []
    current_source = source_lang
    for next_target in path:
        model_name = TRANSLATION_MODELS.get((current_source, next_target))
        if not model_name:
            raise ValueError(f"Brak obsługiwanej pary tłumaczenia: {current_source}->{next_target}")
        prefix = TRANSLATION_PREFIXES.get((current_source, next_target), "")
        current_text = translate_with_model(current_text, model_name, prefix)
        used_models.append(model_name)
        current_source = next_target

    result = TranslationResult(source_lang, target_lang, current_text, used_models)
    save_translation(text, result)
    return result


def translation_path(source_lang: str, target_lang: str) -> list[str]:
    if (source_lang, target_lang) in TRANSLATION_MODELS:
        return [target_lang]
    if (source_lang, "en") in TRANSLATION_MODELS and ("en", target_lang) in TRANSLATION_MODELS:
        return ["en", target_lang]
    raise ValueError(f"Nieobsługiwana para języków: {source_lang}->{target_lang}")


def translate_with_model(text: str, model_name: str, prefix: str = "") -> str:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Brakuje transformers/sentencepiece/sacremoses. Uruchom pip install -r requirements.txt") from exc

    if model_name not in TRANSLATOR_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        TRANSLATOR_CACHE[model_name] = (tokenizer, model)
    tokenizer, model = TRANSLATOR_CACHE[model_name]

    source_text = f"{prefix} {text}".strip()
    encoded = tokenizer([source_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    generated = model.generate(**encoded, max_new_tokens=256)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def save_translation(source_text: str, result: TranslationResult) -> None:
    ensure_lab4_dirs()
    exists = LAB4_TRANSLATIONS_PATH.exists()
    with LAB4_TRANSLATIONS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["source_lang", "target_lang", "source_text", "translation", "models"])
        writer.writerow(
            [result.source_lang, result.target_lang, source_text, result.translation, " | ".join(result.models)]
        )
