from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from lab5_config import OLLAMA_CHAT_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
from lab5_history import save_tool_event
from lab5_tools import TOOL_SCHEMAS, tool_registry


@dataclass
class ToolCallResult:
    tool: str
    arguments: dict[str, Any]
    result: str


@dataclass
class AskResult:
    answer: str
    tool_results: list[ToolCallResult]
    model: str

    def as_message(self) -> str:
        lines = [f"Model: {self.model}"]
        if self.tool_results:
            lines.append("Użyte narzędzia:")
            for item in self.tool_results:
                lines.append(f"- {item.tool}({json.dumps(item.arguments, ensure_ascii=False)})")
        else:
            lines.append("Użyte narzędzia: brak")
        lines.append("")
        lines.append(self.answer)
        return "\n".join(lines)


def ask_with_tools(prompt: str, model: str = OLLAMA_MODEL) -> AskResult:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Pytanie nie może być puste.")

    messages = [
        {
            "role": "system",
            "content": (
                "Jesteś asystentem NLP. Jeśli pytanie wymaga aktualnych danych, obliczeń, pogody, "
                "lokalnej wiedzy albo obrazu, użyj dostępnych narzędzi. Odpowiadaj po polsku."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    first = ollama_chat(model, messages, tools=TOOL_SCHEMAS)
    message = first.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        heuristic_calls = guess_tools(prompt)
        if heuristic_calls:
            tool_results = execute_tool_calls(heuristic_calls, source="heuristic")
            final_answer = synthesize_answer(model, prompt, tool_results)
            return AskResult(final_answer, tool_results, model)
        return AskResult(message.get("content", "").strip() or "Model nie zwrócił odpowiedzi.", [], model)

    tool_results = execute_tool_calls(tool_calls, source="ollama")
    final_messages = messages + [message]
    for item in tool_results:
        final_messages.append(
            {
                "role": "tool",
                "content": item.result,
                "name": item.tool,
            }
        )
    final = ollama_chat(model, final_messages)
    answer = final.get("message", {}).get("content", "").strip()
    if not answer:
        answer = synthesize_answer(model, prompt, tool_results)
    answer = ensure_tool_results_visible(answer, tool_results)
    return AskResult(answer, tool_results, model)


def ollama_chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    return response.json()


def execute_tool_calls(tool_calls: list[Any], source: str) -> list[ToolCallResult]:
    registry = tool_registry()
    results: list[ToolCallResult] = []
    for call in tool_calls:
        name, arguments = normalize_tool_call(call)
        if name not in registry:
            continue
        try:
            result = registry[name](**arguments)
        except Exception as exc:
            result = f"Błąd narzędzia {name}: {exc}"
        save_tool_event(name, arguments, result, source=source)
        results.append(ToolCallResult(name, arguments, str(result)))
    return results


def normalize_tool_call(call: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(call, dict):
        function = call.get("function", call)
        name = function.get("name", call.get("name", ""))
        arguments = function.get("arguments", call.get("arguments", {}))
    else:
        function = getattr(call, "function", call)
        name = getattr(function, "name", "")
        arguments = getattr(function, "arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return str(name), dict(arguments or {})


def synthesize_answer(model: str, prompt: str, tool_results: list[ToolCallResult]) -> str:
    context = "\n\n".join(
        f"{item.tool}({json.dumps(item.arguments, ensure_ascii=False)}):\n{item.result}"
        for item in tool_results
    )
    messages = [
        {
            "role": "system",
            "content": "Na podstawie wyników narzędzi odpowiedz po polsku, krótko i konkretnie.",
        },
        {
            "role": "user",
            "content": f"Pytanie użytkownika:\n{prompt}\n\nWyniki narzędzi:\n{context}",
        },
    ]
    try:
        response = ollama_chat(model, messages)
        return response.get("message", {}).get("content", "").strip() or context
    except Exception:
        return context


def ensure_tool_results_visible(answer: str, tool_results: list[ToolCallResult]) -> str:
    if not tool_results:
        return answer

    answer_lower = answer.lower()
    missing = []
    for item in tool_results:
        probe = item.result[:40].lower()
        if probe and probe not in answer_lower:
            missing.append(item)

    if not missing:
        return answer

    lines = [answer.strip(), "", "Wyniki narzędzi:"]
    for item in missing:
        compact = item.result.replace("\n", " ")
        lines.append(f"- {item.tool}: {compact[:700]}")
    return "\n".join(lines).strip()


def guess_tools(prompt: str) -> list[dict[str, Any]]:
    lower = prompt.lower()
    calls: list[dict[str, Any]] = []

    if any(word in lower for word in ["pogoda", "weather", "temperatura", "spacer"]):
        cities = extract_cities(prompt)
        for city in cities or ["Warszawa"]:
            calls.append({"function": {"name": "get_weather", "arguments": {"city": city}}})

    if any(word in lower for word in ["kto jest", "ceo", "wikipedia", "wyszukaj", "aktualne", "typowa"]):
        calls.append({"function": {"name": "web_search", "arguments": {"query": prompt}}})

    expression = extract_expression(prompt)
    if expression:
        calls.append({"function": {"name": "simple_calculator", "arguments": {"expression": expression}}})

    if any(word in lower for word in ["projekt", "laboratorium", "lab 5", "ollama"]):
        calls.append({"function": {"name": "local_knowledge", "arguments": {"query": prompt}}})

    return calls


def extract_cities(text: str) -> list[str]:
    known = {
        "warszawie": "Warszawa",
        "warszawa": "Warszawa",
        "warsaw": "Warsaw",
        "paryżu": "Paris",
        "paryz": "Paris",
        "paris": "Paris",
        "berlinie": "Berlin",
        "berlin": "Berlin",
        "londynie": "London",
        "london": "London",
    }
    lower = text.lower()
    cities = [city for key, city in known.items() if key in lower]
    return list(dict.fromkeys(cities))


def extract_expression(text: str) -> str:
    match = re.search(r"([-+*/().\d\s]+(?:\*\*|[-+*/])[-+*/().\d\s]+)", text)
    return match.group(1).strip() if match else ""
