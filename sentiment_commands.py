from __future__ import annotations

import asyncio
import shlex

from telegram import Update
from telegram.ext import ContextTypes

from sentiment_compare import compare_sentiment_methods
from sentiment_config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_MAX_LEN, SENTIMENT_METHODS
from sentiment_datasets import add_custom_record, ensure_custom_dataset
from sentiment_methods import predict_sentiment
from sentiment_training import list_saved_models, train_sequence_model


def parse_params(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for token in shlex.split(text)[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        params[key.lower().strip()] = value.strip()
    return params


def help_text() -> str:
    return (
        "Komendy bota:\n\n"
        "Lab 1:\n"
        '/task tokenize "tekst" "neutralny"\n'
        '/full_pipeline "tekst" "neutralny"\n'
        '/classifier "tekst"\n'
        "/stats\n\n"
        "Lab 2:\n"
        "/classify dataset=20news_group method=logreg gridsearch=false run=1\n\n"
        "Lab 3:\n"
        '/sentiment method=rule text="To był świetny film"\n'
        '/sentiment method=nb dataset=custom text="Produkt jest fatalny"\n'
        "/train model=lstm dataset=custom epochs=10 max_len=200\n"
        "/compare dataset=custom methods=rule,nb,rf\n"
        '/add_sentiment "Obsługa była poprawna" "neutralny"\n'
        "/models\n\n"
        "Lab 4:\n"
        '/ner method=spacy text="Steve Jobs założył Apple."\n'
        '/ner method=stanza text="Elon Musk pracuje nad xAI w Austin."\n'
        '/nel text="Steve Jobs" language=en\n'
        '/ned entity="Apple" context="Steve Jobs założył Apple w Kalifornii" language=en\n'
        '/translate text="The quick brown fox jumps over the lazy dog" target_lang=pl\n'
        '/summarize text="długi tekst" summary_type=abstractive length=medium\n'
        '/analyze_entities text="Elon Musk posiada firmę Tesla w Austin." link=true\n'
        '/knowledge_graph text="Elon Musk posiada firmę Tesla w Austin."\n'
        '/language_detect text="To jest przykładowy tekst"\n\n'
        "Metody sentymentu:\n"
        f"{', '.join(SENTIMENT_METHODS)}"
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg:
        await msg.reply_text(help_text())


async def sentiment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        params = parse_params(msg.text)
        method = params.get("method", "rule")
        dataset = params.get("dataset", "custom")
        text = params.get("text")
        if not text:
            raise ValueError('Podaj tekst, np. /sentiment method=rule text="To był świetny film"')

        prediction = await asyncio.to_thread(predict_sentiment, method, text, dataset)
        await msg.reply_text(prediction.as_message())
    except Exception as exc:
        await msg.reply_text(f"Błąd /sentiment: {exc}")


async def train_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        params = parse_params(msg.text)
        model = params.get("model")
        dataset = params.get("dataset", "custom")
        if not model:
            raise ValueError("Podaj model: simplernn, lstm albo gru.")

        epochs = int(params.get("epochs", DEFAULT_EPOCHS))
        batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
        max_len = int(params.get("max_len", DEFAULT_MAX_LEN))

        await msg.reply_text(
            f"Start treningu: model={model}, dataset={dataset}, epochs={epochs}, max_len={max_len}. "
            "To może potrwać kilka minut."
        )
        result = await asyncio.to_thread(
            train_sequence_model,
            model,
            dataset,
            epochs,
            batch_size,
            max_len,
        )
        await msg.reply_text("Trening zakończony.\n\n" + result.as_message())
    except Exception as exc:
        await msg.reply_text(f"Błąd /train: {exc}")


async def compare_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        params = parse_params(msg.text)
        dataset = params.get("dataset", "custom")
        methods_value = params.get("methods", "rule,nb,rf")
        methods = [method.strip().lower() for method in methods_value.split(",") if method.strip()]

        await msg.reply_text(f"Porównuję metody dla dataset={dataset}: {', '.join(methods)}.")
        results, notes, files = await asyncio.to_thread(compare_sentiment_methods, dataset, methods)

        if results.empty:
            response = "Nie udało się policzyć żadnej metody."
        else:
            response_lines = ["Wyniki porównania:"]
            for row in results.sort_values("macro_f1", ascending=False).itertuples(index=False):
                response_lines.append(
                    f"- {row.method}: acc={row.accuracy:.3f}, macro_f1={row.macro_f1:.3f}"
                )
            response_lines.append("")
            response_lines.append("Pliki:")
            response_lines.extend(f"- {path}" for path in files[:8])
            response = "\n".join(response_lines)

        if notes:
            response += "\n\nUwagi:\n" + "\n".join(f"- {note}" for note in notes)
        await msg.reply_text(response[:3900])
    except Exception as exc:
        await msg.reply_text(f"Błąd /compare: {exc}")


async def add_sentiment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        parts = shlex.split(msg.text)
        if len(parts) < 3:
            raise ValueError('/add_sentiment "tekst" "etykieta"')
        count = add_custom_record(parts[1], parts[2])
        await msg.reply_text(f"Dopisano rekord do sentiment_dataset.csv. Liczba rekordów: {count}")
    except Exception as exc:
        await msg.reply_text(f"Błąd /add_sentiment: {exc}")


async def models_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return

    ensure_custom_dataset()
    rows = list_saved_models()
    if not rows:
        await msg.reply_text("Brak zapisanych modeli w katalogu models/. Użyj najpierw /train.")
        return

    lines = ["Zapisane modele:"]
    for row in rows:
        lines.append(
            f"- {row['model']} / {row['dataset']}: {row['path']} "
            f"(tokenizer: {row['tokenizer']}, encoder: {row['encoder']})"
        )
    await msg.reply_text("\n".join(lines))
