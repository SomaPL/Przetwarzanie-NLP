from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from wordcloud import WordCloud

from sentiment_config import LAB3_PLOTS_DIR, SENTIMENT_LABELS


def ensure_plot_dir() -> None:
    LAB3_PLOTS_DIR.mkdir(exist_ok=True)


def plot_training_history(history: dict, model_name: str, dataset: str) -> str:
    ensure_plot_dir()
    path = LAB3_PLOTS_DIR / f"train_history_{model_name}_{dataset}.png"

    plt.figure(figsize=(9, 5))
    if "accuracy" in history:
        plt.plot(history["accuracy"], label="accuracy")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val_accuracy")
    if "loss" in history:
        plt.plot(history["loss"], label="loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.title(f"Historia uczenia: {model_name} / {dataset}")
    plt.xlabel("Epoka")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def plot_confusion(y_true: list[str], y_pred: list[str], method: str, dataset: str) -> str:
    ensure_plot_dir()
    labels = [label for label in SENTIMENT_LABELS if label in set(y_true) | set(y_pred)]
    labels = labels or sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    path = LAB3_PLOTS_DIR / f"confusion_{method}_{dataset}.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, colorbar=False)
    plt.title(f"Macierz pomyłek: {method} / {dataset}")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_compare_results(results: pd.DataFrame, dataset: str) -> str:
    ensure_plot_dir()
    path = LAB3_PLOTS_DIR / f"compare_methods_{dataset}.png"
    data = results.sort_values("macro_f1", ascending=False)

    plt.figure(figsize=(9, 5))
    try:
        import seaborn as sns

        sns.barplot(data=data, x="method", y="macro_f1")
    except ImportError:
        plt.bar(data["method"], data["macro_f1"])
    plt.ylim(0, 1)
    plt.ylabel("macro_f1")
    plt.title(f"Porównanie metod sentymentu: {dataset}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def generate_sentiment_wordclouds(df: pd.DataFrame, dataset: str) -> list[str]:
    ensure_plot_dir()
    paths: list[str] = []
    for label, group in df.groupby("label"):
        path = LAB3_PLOTS_DIR / f"wordcloud_{label}.png"
        text = " ".join(group["text"].astype(str).tolist()) or "brak danych"
        cloud = WordCloud(width=1200, height=800, background_color="white").generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word cloud: {label} ({dataset})")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths.append(str(path))
    return paths


def plot_class_distribution(df: pd.DataFrame, dataset: str) -> str:
    ensure_plot_dir()
    path = LAB3_PLOTS_DIR / f"class_distribution_{dataset}.png"
    counts = df["label"].value_counts().reindex(SENTIMENT_LABELS).dropna()
    plt.figure(figsize=(7, 5))
    try:
        import seaborn as sns

        sns.barplot(x=counts.index, y=counts.values)
    except ImportError:
        plt.bar(counts.index, counts.values)
    plt.title(f"Rozkład klas: {dataset}")
    plt.ylabel("Liczba przykładów")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)
