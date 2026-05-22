from __future__ import annotations

import asyncio
import shlex

from telegram import Update
from telegram.ext import ContextTypes

from lab5_history import read_tool_history
from lab5_ollama import ask_with_tools
from lab5_tools import TOOL_SCHEMAS, analyze_image, get_weather, local_knowledge, simple_calculator, web_search


def parse_params(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for token in shlex.split(text)[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        params[key.lower().strip()] = value.strip()
    return params


def parse_free_text_command(text: str) -> str:
    parts = shlex.split(text)
    if len(parts) <= 1:
        return ""
    if len(parts) == 2 and "=" not in parts[1]:
        return parts[1]
    params = parse_params(text)
    return params.get("text", params.get("query", ""))


async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        prompt = parse_free_text_command(msg.text)
        if not prompt:
            raise ValueError('/ask "Czy dziś jest dobra pogoda na spacer w Warszawie?"')
        await msg.reply_text("Myślę i dobieram narzędzia...")
        result = await asyncio.to_thread(ask_with_tools, prompt)
        await msg.reply_text(result.as_message()[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /ask: {exc}")


async def tools_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    lines = ["Dostępne tools:"]
    for tool in TOOL_SCHEMAS:
        fn = tool["function"]
        lines.append(f"- {fn['name']}: {fn['description']}")
    await msg.reply_text("\n".join(lines))


async def tool_history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    records = read_tool_history(10)
    if not records:
        await msg.reply_text("Historia tooli jest pusta.")
        return
    lines = ["Ostatnie wywołania tooli:"]
    for record in records:
        lines.append(f"- {record['timestamp']} {record['tool']} {record['arguments']}")
    await msg.reply_text("\n".join(lines)[:3900])


async def weather_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        city = params.get("city") or parse_free_text_command(msg.text)
        result = await asyncio.to_thread(get_weather, city)
        await msg.reply_text(result)
    except Exception as exc:
        await msg.reply_text(f"Błąd /weather: {exc}")


async def web_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        query = parse_params(msg.text).get("query") or parse_free_text_command(msg.text)
        result = await asyncio.to_thread(web_search, query)
        await msg.reply_text(result[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /web: {exc}")


async def calc_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        expression = parse_params(msg.text).get("expression") or parse_free_text_command(msg.text)
        result = simple_calculator(expression)
        await msg.reply_text(result)
    except Exception as exc:
        await msg.reply_text(f"Błąd /calc: {exc}")


async def knowledge_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        query = parse_params(msg.text).get("query") or parse_free_text_command(msg.text)
        result = local_knowledge(query)
        await msg.reply_text(result[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /knowledge: {exc}")


async def vision_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        image_path = params.get("image_path", "")
        prompt = params.get("prompt", "Describe this image in detail.")
        result = await asyncio.to_thread(analyze_image, image_path, prompt)
        await msg.reply_text(result[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /vision: {exc}")

