import asyncio
import os
import shlex
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from classifier import predict_text_class
from lab4_commands import (
    analyze_entities_handler,
    knowledge_graph_handler,
    language_detect_handler,
    ned_handler,
    nel_handler,
    ner_handler,
    summarize_handler,
    translate_handler,
)
from lab5_commands import (
    ask_handler,
    calc_handler,
    knowledge_handler,
    tool_history_handler,
    tools_handler,
    vision_handler,
    weather_handler,
    web_handler,
)
from lab2_experiments import parse_classify_command, run_dataset_experiment
from sentiment_commands import (
    add_sentiment_handler,
    compare_handler,
    help_handler,
    models_handler,
    sentiment_handler,
    train_handler,
)
from nlp import (
    bag_of_words,
    clean_text,
    generate_ngrams,
    lemmatize_tokens,
    remove_stopwords,
    split_sentences,
    stem_tokens,
    text_stats,
    tfidf_features,
    tokenize,
)
from plots import (
    plot_class_distribution,
    plot_histogram,
    plot_top_words,
    plot_wordcloud,
)
from storage import append_sentence_record, load_sentences

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")


def parse_quoted_args(text: str) -> list[str]:
    return shlex.split(text)


async def send_photos(update: Update, paths: list[str]) -> None:
    if not update.effective_message:
        return
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                await update.effective_message.reply_photo(photo=f)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    await update.effective_message.reply_text(
        "Cześć! Dostępne komendy:\n\n"
        '/task tokenize "To był bardzo interesujący wykład." "neutralny"\n'
        '/task remove_stopwords "To był bardzo interesujący wykład." "neutralny"\n'
        '/task lemmatize "To był bardzo interesujący wykład." "neutralny"\n'
        '/task stemming "To był bardzo interesujący wykład." "neutralny"\n'
        '/task stats "To był bardzo interesujący wykład." "neutralny"\n'
        '/task n-grams "To był bardzo interesujący wykład." "neutralny"\n'
        '/task plot_histogram "To był bardzo interesujący wykład." "neutralny"\n'
        '/task plot_wordcloud "To był bardzo interesujący wykład." "neutralny"\n'
        '/task plot_top_words "To był bardzo interesujący wykład." "neutralny"\n\n'
        '/full_pipeline "System działa szybko, ale interfejs wymaga poprawy." "neutralny"\n'
        '/classifier "To był fantastyczny film"\n'
        '/classify dataset=20news_group method=logreg gridsearch=false run=1\n'
        '/sentiment method=rule text="To był świetny film"\n'
        "/train model=lstm dataset=custom epochs=10 max_len=200\n"
        "/compare dataset=custom methods=rule,nb,rf\n"
        '/add_sentiment "Obsługa była poprawna" "neutralny"\n'
        "/models\n"
        "/help\n"
        '/ner method=spacy text="Steve Jobs założył Apple."\n'
        '/nel text="Steve Jobs" language=en\n'
        '/translate text="The quick brown fox jumps over the lazy dog" target_lang=pl\n'
        '/summarize text="Długi tekst..." summary_type=abstractive length=medium\n'
        '/ask "Czy dziś jest dobra pogoda na spacer w Warszawie?"\n'
        '/weather city="Warszawa"\n'
        '/web query="CEO Tesli"\n'
        '/calc expression="2+2*5"\n'
        '/tools\n'
        "/stats"
    )


async def task_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        parts = parse_quoted_args(msg.text)
    except ValueError:
        await msg.reply_text('Błędna składnia. Użyj: /task <zadanie> "tekst" "klasa"')
        return

    if len(parts) < 4:
        await msg.reply_text('Za mało argumentów. Użyj: /task <zadanie> "tekst" "klasa"')
        return

    _, task_name, text, label = parts[0], parts[1], parts[2], parts[3]
    task_name = task_name.lower().strip()

    if not text.strip():
        await msg.reply_text("Tekst nie może być pusty.")
        return

    tokens = tokenize(text)
    cleaned = clean_text(text)
    clean_tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(clean_tokens)

    response = ""
    plot_paths = []

    try:
        if task_name == "tokenize":
            response = f"Tokeny:\n{tokens}"

        elif task_name == "remove_stopwords":
            response = f"Po usunięciu stopwords:\n{tokens_no_stop}"

        elif task_name == "lemmatize":
            lemmas = lemmatize_tokens(tokens_no_stop)
            response = f"Lematy:\n{lemmas}"

        elif task_name == "stemming":
            stems = stem_tokens(tokens_no_stop)
            response = f"Stemming:\n{stems}"

        elif task_name == "stats":
            stats = text_stats(tokens_no_stop)
            response = (
                "Statystyki tekstu:\n"
                f"Liczba tokenów: {stats['token_count']}\n"
                f"Liczba unikalnych tokenów: {stats['unique_tokens']}\n"
                f"Średnia długość tokenu: {stats['avg_token_length']}\n"
                f"Najczęstsze słowa: {stats['top_words']}"
            )

        elif task_name == "n-grams":
            bigrams = generate_ngrams(tokens_no_stop, 2)
            trigrams = generate_ngrams(tokens_no_stop, 3)
            response = f"2-gramy:\n{bigrams}\n\n3-gramy:\n{trigrams}"

        elif task_name == "plot_histogram":
            path = plot_histogram(tokens_no_stop)
            plot_paths.append(path)
            response = f"Wygenerowano histogram:\n{path}"

        elif task_name == "plot_wordcloud":
            path = plot_wordcloud(tokens_no_stop)
            plot_paths.append(path)
            response = f"Wygenerowano wordcloud:\n{path}"

        elif task_name == "plot_top_words":
            path = plot_top_words(tokens_no_stop)
            plot_paths.append(path)
            response = f"Wygenerowano wykres najczęstszych słów:\n{path}"

        else:
            await msg.reply_text(
                "Nieznane zadanie.\n"
                "Dostępne: tokenize, remove_stopwords, lemmatize, stemming, "
                "stats, n-grams, plot_histogram, plot_wordcloud, plot_top_words"
            )
            return

        append_sentence_record(text=text, label=label)
        await msg.reply_text(response)

        if plot_paths:
            await send_photos(update, plot_paths)

    except Exception as e:
        await msg.reply_text(f"Wystąpił błąd podczas wykonywania zadania: {e}")


async def full_pipeline_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        parts = parse_quoted_args(msg.text)
    except ValueError:
        await msg.reply_text('Błędna składnia. Użyj: /full_pipeline "tekst" "klasa"')
        return

    if len(parts) < 3:
        await msg.reply_text('Za mało argumentów. Użyj: /full_pipeline "tekst" "klasa"')
        return

    _, text, label = parts[0], parts[1], parts[2]

    if not text.strip():
        await msg.reply_text("Tekst nie może być pusty.")
        return

    try:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        tokens_no_stop = remove_stopwords(tokens)
        lemmas = lemmatize_tokens(tokens_no_stop)
        stems = stem_tokens(lemmas)
        bow = bag_of_words([" ".join(stems)])
        tfidf = tfidf_features([" ".join(stems)])
        stats = text_stats(stems)

        plot_paths = [
            plot_top_words(stems),
            plot_histogram(stems),
            plot_wordcloud(stems),
        ]

        sentences = split_sentences(text)
        if len(sentences) > 1:
            for sentence in sentences:
                append_sentence_record(text=sentence, label=label)
        else:
            append_sentence_record(text=text, label=label)

        response = (
            "Pełny pipeline wykonany.\n\n"
            f"1. Clean text:\n{cleaned}\n\n"
            f"2. Tokeny:\n{tokens}\n\n"
            f"3. Po usunięciu stopwords:\n{tokens_no_stop}\n\n"
            f"4. Lematy:\n{lemmas}\n\n"
            f"5. Stemming:\n{stems}\n\n"
            f"6. Bag of Words:\n{bow}\n\n"
            f"7. TF-IDF:\n{tfidf}\n\n"
            f"8. Statystyki:\n"
            f"- liczba tokenów: {stats['token_count']}\n"
            f"- liczba unikalnych tokenów: {stats['unique_tokens']}\n"
            f"- średnia długość tokenu: {stats['avg_token_length']}\n"
            f"- najczęstsze słowa: {stats['top_words']}"
        )

        await msg.reply_text(response)
        await send_photos(update, plot_paths)

    except Exception as e:
        await msg.reply_text(f"Błąd w /full_pipeline: {e}")


async def classifier_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        parts = parse_quoted_args(msg.text)
    except ValueError:
        await msg.reply_text('Błędna składnia. Użyj: /classifier "tekst"')
        return

    if len(parts) < 2:
        await msg.reply_text('Za mało argumentów. Użyj: /classifier "tekst"')
        return

    _, text = parts[0], parts[1]

    if not text.strip():
        await msg.reply_text("Tekst nie może być pusty.")
        return

    try:
        predicted = predict_text_class(text)
        await msg.reply_text(f"Przewidziana klasa: {predicted}")
    except Exception as e:
        await msg.reply_text(f"Błąd klasyfikacji: {e}")


async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return

    try:
        records = load_sentences()
        if not records:
            await msg.reply_text("Brak danych w sentences.json")
            return

        all_texts = [r["text"] for r in records]
        all_labels = [r["class"] for r in records]

        merged_text = " ".join(all_texts)
        tokens = remove_stopwords(tokenize(clean_text(merged_text)))

        stats = text_stats(tokens)
        bigrams = generate_ngrams(tokens, 2)
        trigrams = generate_ngrams(tokens, 3)

        plot_paths = [
            plot_top_words(tokens),
            plot_histogram(tokens),
            plot_wordcloud(tokens),
            plot_class_distribution(all_labels),
        ]

        response = (
            "Statystyki całego zbioru:\n\n"
            f"Liczba rekordów: {len(records)}\n"
            f"Unikalne tokeny: {stats['unique_token_list']}\n\n"
            f"Unikalne 2-gramy: {list(dict.fromkeys(bigrams))[:20]}\n\n"
            f"Unikalne 3-gramy: {list(dict.fromkeys(trigrams))[:20]}\n\n"
            f"Najczęstsze słowa: {stats['top_words']}\n"
            f"Liczność klas: {stats['class_counts_from_labels'](all_labels)}"
        )

        await msg.reply_text(response)
        await send_photos(update, plot_paths)

    except Exception as e:
        await msg.reply_text(f"Błąd w /stats: {e}")


async def classify_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg or not msg.text:
        return

    try:
        config = parse_classify_command(msg.text)
    except Exception as e:
        await msg.reply_text(f"Błędna składnia /classify: {e}")
        return

    await msg.reply_text(
        "Startuję eksperyment Lab 2. To może potrwać kilka minut, "
        "szczególnie dla method=all albo gridsearch=true."
    )

    try:
        report = await asyncio.to_thread(run_dataset_experiment, config)
        await msg.reply_text(report.as_message())
    except Exception as e:
        await msg.reply_text(f"Błąd w /classify: {e}")


def main() -> None:
    if not TOKEN:
        raise RuntimeError("Brak tokena bota. Dodaj BOT_TOKEN do pliku .env")

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("task", task_handler))
    app.add_handler(CommandHandler("full_pipeline", full_pipeline_handler))
    app.add_handler(CommandHandler("classifier", classifier_handler))
    app.add_handler(CommandHandler("stats", stats_handler))
    app.add_handler(CommandHandler("classify", classify_handler))
    app.add_handler(CommandHandler("sentiment", sentiment_handler))
    app.add_handler(CommandHandler("train", train_handler))
    app.add_handler(CommandHandler("compare", compare_handler))
    app.add_handler(CommandHandler("add_sentiment", add_sentiment_handler))
    app.add_handler(CommandHandler("models", models_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("ner", ner_handler))
    app.add_handler(CommandHandler("nel", nel_handler))
    app.add_handler(CommandHandler("ned", ned_handler))
    app.add_handler(CommandHandler("translate", translate_handler))
    app.add_handler(CommandHandler("summarize", summarize_handler))
    app.add_handler(CommandHandler("analyze_entities", analyze_entities_handler))
    app.add_handler(CommandHandler("knowledge_graph", knowledge_graph_handler))
    app.add_handler(CommandHandler("language_detect", language_detect_handler))
    app.add_handler(CommandHandler("ask", ask_handler))
    app.add_handler(CommandHandler("tools", tools_handler))
    app.add_handler(CommandHandler("tool_history", tool_history_handler))
    app.add_handler(CommandHandler("weather", weather_handler))
    app.add_handler(CommandHandler("web", web_handler))
    app.add_handler(CommandHandler("calc", calc_handler))
    app.add_handler(CommandHandler("knowledge", knowledge_handler))
    app.add_handler(CommandHandler("vision", vision_handler))

    print("Bot działa...")
    app.run_polling()


if __name__ == "__main__":
    main()
