from __future__ import annotations

import asyncio
import shlex

from telegram import Update
from telegram.ext import ContextTypes

from lab4_graph import build_knowledge_graph
from lab4_language import detect_language
from lab4_linking import disambiguate_entity, link_entity, save_link_candidates
from lab4_ner import run_ner
from lab4_summarization import summarize_text
from lab4_translation import translate_text


def parse_params(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for token in shlex.split(text)[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        params[key.lower().strip()] = value.strip()
    return params


async def ner_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        method = params.get("method", "spacy")
        text = params.get("text", "")
        language = params.get("language")
        result = await asyncio.to_thread(run_ner, method, text, language)
        await msg.reply_text(result.as_message()[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /ner: {exc}")


async def nel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        entity = params.get("text", "")
        language = params.get("language", "en")
        candidates = await asyncio.to_thread(link_entity, entity, language, "")
        save_link_candidates(entity, language, "", candidates)
        await msg.reply_text(format_candidates(entity, candidates)[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /nel: {exc}")


async def ned_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        entity = params.get("entity", "")
        context_text = params.get("context", params.get("text", ""))
        language = params.get("language", "en")
        best = await asyncio.to_thread(disambiguate_entity, entity, context_text, language)
        if best is None:
            await msg.reply_text("Nie znaleziono wystarczająco pewnej interpretacji.")
        else:
            save_link_candidates(entity, language, context_text, [best])
            await msg.reply_text("\n".join([f"Entity: {entity}", "Best candidate:", *best.as_lines(1)]))
    except Exception as exc:
        await msg.reply_text(f"Błąd /ned: {exc}")


async def translate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        text = params.get("text", "")
        target_lang = params.get("target_lang", params.get("target", "pl"))
        source_lang = params.get("source_lang")
        result = await asyncio.to_thread(translate_text, text, target_lang, source_lang)
        await msg.reply_text(result.as_message()[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /translate: {exc}")


async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        text = params.get("text", "")
        summary_type = params.get("summary_type", "abstractive")
        length = params.get("length", "medium")
        model = params.get("model")
        custom_prompt = params.get("prompt", "")
        if model:
            result = await asyncio.to_thread(summarize_text, text, summary_type, length, model, custom_prompt)
        else:
            result = await asyncio.to_thread(summarize_text, text, summary_type, length, custom_prompt=custom_prompt)
        await msg.reply_text(result.as_message()[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /summarize: {exc}")


async def analyze_entities_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        text = params.get("text", "")
        method = params.get("method", "spacy")
        language = params.get("language")
        link = params.get("link", "true").lower() == "true"
        result = await asyncio.to_thread(run_ner, method, text, language)

        lines = ["ENTITIES FOUND:"]
        for entity in result.entities:
            lines.append(f"- {entity.text} ({entity.label}) [{entity.start}:{entity.end}]")
            if link:
                candidates = link_entity(entity.text, result.language, text)
                save_link_candidates(entity.text, result.language, text, candidates[:3])
                if candidates:
                    best = candidates[0]
                    lines.append(f"  Wikidata: {best.wikidata_id}")
                    lines.append(f"  Wikipedia: {best.wikipedia_url or 'Not found'}")
                else:
                    lines.append("  Wikidata: Not found")
        if not result.entities:
            lines.append("- brak encji")
        await msg.reply_text("\n".join(lines)[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /analyze_entities: {exc}")


async def knowledge_graph_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        text = params.get("text", "")
        method = params.get("method", "spacy")
        language = params.get("language")
        result = await asyncio.to_thread(build_knowledge_graph, text, method, language)
        await msg.reply_text(result.as_message())
    except Exception as exc:
        await msg.reply_text(f"Błąd /knowledge_graph: {exc}")


async def language_detect_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return
    try:
        params = parse_params(msg.text)
        result = detect_language(params.get("text", ""))
        await msg.reply_text(result.as_message())
    except Exception as exc:
        await msg.reply_text(f"Błąd /language_detect: {exc}")


def format_candidates(entity: str, candidates) -> str:
    lines = [f"Entity: {entity}", "Candidates:"]
    if not candidates:
        lines.append("- brak kandydatów")
        return "\n".join(lines)
    for index, candidate in enumerate(candidates, start=1):
        lines.extend(candidate.as_lines(index))
    return "\n".join(lines)

