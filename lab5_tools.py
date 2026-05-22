from __future__ import annotations

import ast
import base64
import json
import math
import operator
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from lab5_config import (
    FALLBACK_CITY_COORDS,
    LOCAL_KNOWLEDGE_PATH,
    OLLAMA_GENERATE_URL,
    OLLAMA_TIMEOUT,
    OLLAMA_VISION_MODEL,
    WEATHER_CODE_DESCRIPTIONS,
)
from lab5_history import save_tool_event


HEADERS = {
    "User-Agent": "Przetwarzanie-NLP-Lab5/1.0 (student project; Telegram NLP bot)"
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search internet sources such as Wikipedia and Wikidata and return short summary results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return current weather for a city using Open-Meteo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local_knowledge",
            "description": "Search local project knowledge stored in JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Local knowledge query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Describe an image using a local Ollama vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image file"},
                    "prompt": {"type": "string", "description": "Optional vision prompt"},
                },
                "required": ["image_path"],
            },
        },
    },
]


def web_search(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query nie może być pusty.")

    lines = [f"Web search: {query}"]
    direct_answer = wikidata_direct_answer(query)
    if direct_answer:
        lines.append("Direct answer:")
        lines.append(direct_answer)

    wiki_results = wikipedia_search(query)
    if wiki_results:
        lines.append("Wikipedia:")
        for index, item in enumerate(wiki_results, start=1):
            lines.append(f"{index}. {item['title']} - {item['description']}")
            lines.append(f"   {item['url']}")

    wikidata_results = wikidata_search(query)
    if wikidata_results:
        lines.append("Wikidata:")
        for index, item in enumerate(wikidata_results, start=1):
            lines.append(f"{index}. {item['label']} ({item['id']}) - {item['description']}")

    result = "\n".join(lines) if len(lines) > 1 else "Brak wyników."
    save_tool_event("web_search", {"query": query}, result)
    return result


def wikidata_direct_answer(query: str) -> str:
    lower = query.lower()
    if "ceo" not in lower and "prezes" not in lower:
        return ""

    company_query = cleanup_company_query(query)
    if not company_query:
        return ""

    entities = wikidata_search(company_query)
    if not entities:
        return ""
    company_id = entities[0]["id"]
    company_label = entities[0]["label"]

    sparql = f"""
    SELECT ?person ?personLabel WHERE {{
      wd:{company_id} wdt:P169 ?person .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pl,en". }}
    }}
    LIMIT 3
    """
    try:
        response = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": sparql, "format": "json"},
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        bindings = response.json().get("results", {}).get("bindings", [])
    except Exception:
        return ""

    if not bindings:
        return ""
    people = [item["personLabel"]["value"] for item in bindings if "personLabel" in item]
    return f"CEO / chief executive officer dla {company_label}: {', '.join(people)}"


def cleanup_company_query(query: str) -> str:
    lower = query.lower()
    replacements = {
        "kto jest": "",
        "kim jest": "",
        "ceo": "",
        "prezesem": "",
        "prezes": "",
        "firmy": "",
        "spółki": "",
        "spolki": "",
        "?": "",
    }
    for old, new in replacements.items():
        lower = lower.replace(old, new)
    lower = lower.strip()
    aliases = {
        "tesli": "Tesla, Inc.",
        "tesla": "Tesla, Inc.",
        "apple": "Apple Inc.",
        "microsoft": "Microsoft",
        "google": "Google",
    }
    return aliases.get(lower, lower)


def wikipedia_search(query: str, language: str = "en") -> list[dict[str, str]]:
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 3,
        "namespace": 0,
        "format": "json",
    }
    response = requests.get(
        f"https://{language}.wikipedia.org/w/api.php",
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    titles = data[1] if len(data) > 1 else []
    descriptions = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []
    return [
        {
            "title": title,
            "description": descriptions[index] if index < len(descriptions) else "",
            "url": urls[index] if index < len(urls) else "",
        }
        for index, title in enumerate(titles)
    ]


def wikidata_search(query: str) -> list[dict[str, str]]:
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "uselang": "en",
        "search": query,
        "limit": 3,
    }
    response = requests.get(
        "https://www.wikidata.org/w/api.php",
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    return [
        {
            "id": item.get("id", ""),
            "label": item.get("label", ""),
            "description": item.get("description", ""),
        }
        for item in response.json().get("search", [])
    ]


def get_weather(city: str) -> str:
    city = city.strip()
    if not city:
        raise ValueError("City nie może być puste.")

    latitude, longitude, display_name = resolve_city(city)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    current = response.json().get("current", {})
    code = int(current.get("weather_code", -1))
    description = WEATHER_CODE_DESCRIPTIONS.get(code, f"weather code {code}")
    result = (
        f"{display_name}: {current.get('temperature_2m')}°C "
        f"(feels like {current.get('apparent_temperature')}°C), "
        f"humidity {current.get('relative_humidity_2m')}%, "
        f"wind {current.get('wind_speed_10m')} km/h, {description}"
    )
    save_tool_event("get_weather", {"city": city}, result)
    return result


def resolve_city(city: str) -> tuple[float, float, str]:
    key = city.lower().strip()
    if key in FALLBACK_CITY_COORDS:
        return FALLBACK_CITY_COORDS[key]

    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params=params,
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"Nie znaleziono miasta: {city}")
    best = results[0]
    return float(best["latitude"]), float(best["longitude"]), best.get("name", city)


ALLOWED_OPERATORS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


def simple_calculator(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        raise ValueError("Expression nie może być puste.")
    tree = ast.parse(expression, mode="eval")
    value = eval_math_node(tree.body)
    result = f"{expression} = {value}"
    save_tool_event("simple_calculator", {"expression": expression}, result)
    return result


def eval_math_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in ALLOWED_NAMES:
        return ALLOWED_NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](eval_math_node(node.left), eval_math_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
        return ALLOWED_UNARY[type(node.op)](eval_math_node(node.operand))
    raise ValueError("Dozwolone są tylko proste działania matematyczne.")


def local_knowledge(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Query nie może być puste.")
    if not LOCAL_KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Brak lokalnej bazy wiedzy: {LOCAL_KNOWLEDGE_PATH}")

    records = json.loads(LOCAL_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    query_words = tokenize(query)
    scored = []
    for record in records:
        haystack = " ".join(
            [record.get("title", ""), record.get("content", ""), " ".join(record.get("tags", []))]
        )
        overlap = len(query_words & tokenize(haystack))
        if overlap:
            scored.append((overlap, record))
    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored:
        result = "Brak dopasowań w lokalnej bazie wiedzy."
    else:
        lines = [f"Local knowledge: {query}"]
        for score, record in scored[:3]:
            lines.append(f"- {record.get('title')}: {record.get('content')} (score={score})")
        result = "\n".join(lines)
    save_tool_event("local_knowledge", {"query": query}, result)
    return result


def analyze_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": OLLAMA_VISION_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json().get("response", "").strip()
    if not result:
        result = "Model vision zwrócił pustą odpowiedź."
    save_tool_event("analyze_image", {"image_path": image_path, "prompt": prompt}, result)
    return result


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"(?u)\b\w\w+\b", text.lower())
        if token not in {"oraz", "jest", "the", "and", "for", "with", "czy", "jak", "what"}
    }


def tool_registry() -> dict[str, Callable[..., str]]:
    return {
        "web_search": web_search,
        "get_weather": get_weather,
        "simple_calculator": simple_calculator,
        "local_knowledge": local_knowledge,
        "analyze_image": analyze_image,
    }
