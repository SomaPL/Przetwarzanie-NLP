import os
from pathlib import Path


LAB4_RESULTS_DIR = Path("lab4results")
LAB4_PLOTS_DIR = Path("lab4plots")
LAB4_ENTITIES_PATH = LAB4_RESULTS_DIR / "lab4_entities.csv"
LAB4_LINKS_PATH = LAB4_RESULTS_DIR / "lab4_entity_links.csv"
LAB4_SUMMARIES_PATH = LAB4_RESULTS_DIR / "lab4_summaries.csv"
LAB4_TRANSLATIONS_PATH = LAB4_RESULTS_DIR / "lab4_translations.csv"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))

SUPPORTED_LANGUAGES = ["pl", "en", "de", "fr", "es"]
TRANSLATION_MODELS = {
    ("pl", "en"): "Helsinki-NLP/opus-mt-pl-en",
    ("en", "pl"): "Helsinki-NLP/opus-mt-en-ine",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-ine",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-ine",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-ine",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
}

TRANSLATION_PREFIXES = {
    ("en", "pl"): ">>pol<<",
    ("en", "de"): ">>deu<<",
    ("en", "fr"): ">>fra<<",
    ("en", "es"): ">>spa<<",
}


def ensure_lab4_dirs() -> None:
    LAB4_RESULTS_DIR.mkdir(exist_ok=True)
    LAB4_PLOTS_DIR.mkdir(exist_ok=True)
