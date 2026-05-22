import os
from pathlib import Path


LAB5_RESULTS_DIR = Path("lab5results")
LAB5_HISTORY_PATH = LAB5_RESULTS_DIR / "tool_history.jsonl"
LOCAL_KNOWLEDGE_PATH = Path("data/local_knowledge.json")

OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_GENERATE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}

FALLBACK_CITY_COORDS = {
    "warszawa": (52.23, 21.01, "Warszawa"),
    "warsaw": (52.23, 21.01, "Warsaw"),
    "kraków": (50.06, 19.94, "Kraków"),
    "krakow": (50.06, 19.94, "Kraków"),
    "paryż": (48.86, 2.35, "Paris"),
    "paris": (48.86, 2.35, "Paris"),
    "berlin": (52.52, 13.41, "Berlin"),
    "london": (51.51, -0.13, "London"),
    "londyn": (51.51, -0.13, "London"),
}


def ensure_lab5_dirs() -> None:
    LAB5_RESULTS_DIR.mkdir(exist_ok=True)
    LOCAL_KNOWLEDGE_PATH.parent.mkdir(exist_ok=True)

