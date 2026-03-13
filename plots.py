import os
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
from wordcloud import WordCloud

PLOTS_DIR = "plots"


def ensure_plots_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def build_plot_path(prefix: str = "Sentence") -> str:
    ensure_plots_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(PLOTS_DIR, f"{prefix}_{timestamp}.png")


def plot_histogram(tokens: list[str]) -> str:
    lengths = [len(t) for t in tokens]
    path = build_plot_path("Sentence")

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=10)
    plt.title("Histogram długości tokenów")
    plt.xlabel("Długość tokenu")
    plt.ylabel("Liczba tokenów")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def plot_wordcloud(tokens: list[str]) -> str:
    text = " ".join(tokens) if tokens else "brak danych"
    path = build_plot_path("Sentence")

    wc = WordCloud(width=1200, height=800, background_color="white").generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def plot_top_words(tokens: list[str], top_n: int = 10) -> str:
    counts = Counter(tokens)
    most_common = counts.most_common(top_n)

    labels = [x[0] for x in most_common] if most_common else ["brak"]
    values = [x[1] for x in most_common] if most_common else [0]

    path = build_plot_path("Sentence")

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title("Najczęstsze słowa")
    plt.xlabel("Słowo")
    plt.ylabel("Liczba wystąpień")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def plot_class_distribution(labels: list[str]) -> str:
    counts = Counter(labels)

    x = list(counts.keys()) if counts else ["brak"]
    y = list(counts.values()) if counts else [0]

    path = build_plot_path("Sentence")

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.title("Liczność klas")
    plt.xlabel("Klasa")
    plt.ylabel("Liczba przykładów")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path